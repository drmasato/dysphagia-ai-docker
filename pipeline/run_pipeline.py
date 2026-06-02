"""
統合パイプラインオーケストレーター
nnU-Net → TotalSegmentator → BIANCA → 予測APIの順に呼び出し、最終レポートを生成する。
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Any

import requests

PREDICTOR_URL = os.environ.get("PREDICTOR_URL", "http://localhost:8000")
TIMEOUT_SEC = 900


def run_container(
    container_name: str,
    command: list[str],
    fallback: dict,
    output_json: str,
) -> dict:
    """Dockerコンテナを実行し、結果JSONを読み込む。エラー時はfallbackを返す。"""
    print(f"\n[Pipeline] Step: {container_name} 実行中...")
    try:
        result = subprocess.run(
            ["docker", "exec", container_name] + command,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        if os.path.exists(output_json):
            with open(output_json, encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"出力JSONが見つかりません: {output_json}")

    except Exception as e:
        print(f"[Pipeline] 警告: {container_name} 失敗（フォールバック値を使用）: {e}")
        return fallback


def run_nnunet(
    patient_id: str,
    input_path: str,
    output_dir: str,
    is_dicom: bool = True,
) -> dict:
    fallback = {
        "patient_id": patient_id,
        "lesion_volume_ml": 0.0,
        "severity": "unknown",
        "hotspot_affected_count": 0,
        "dysphagia_risk_from_lesion": "不明（計測不可）",
    }
    cmd = [
        "python3", "inference.py",
        "--input", input_path,
        "--output", output_dir,
        "--patient-id", patient_id,
    ]
    if is_dicom:
        cmd.append("--dicom")
    output_json = os.path.join(output_dir, f"{patient_id}_nnunet_result.json")
    return run_container("dysphagia-nnunet", cmd, fallback, output_json)


def run_totalseg(
    patient_id: str,
    input_nifti: str,
    output_dir: str,
    sex: str = "male",
) -> dict:
    fallback = {
        "patient_id": patient_id,
        "tongue_csa_cm2": 4.0,
        "superior_pharyngeal_constrictor_volume_cm3": 2.0,
        "overall_assessment": {"overall_risk": "不明（計測不可）"},
    }
    cmd = [
        "python3", "quantify_muscles.py",
        "--input", input_nifti,
        "--output", output_dir,
        "--patient-id", patient_id,
        "--sex", sex,
    ]
    output_json = os.path.join(output_dir, f"{patient_id}_muscles_result.json")
    return run_container("dysphagia-totalseg", cmd, fallback, output_json)


def run_bianca(
    patient_id: str,
    flair_path: str,
    output_dir: str,
    t1_path: str = None,
    age: int = 65,
) -> dict:
    fallback = {
        "patient_id": patient_id,
        "wmh_volume_ml": 0.0,
        "fazekas_grade": 1,
        "dysphagia_risk": "不明（計測不可）",
    }
    cmd = [
        "bash", "bianca_pipeline.sh",
        "--flair", flair_path,
        "--output", output_dir,
        "--patient-id", patient_id,
    ]
    if t1_path:
        cmd += ["--t1", t1_path]
    output_json = os.path.join(output_dir, f"{patient_id}_wmh_result.json")
    return run_container("dysphagia-bianca", cmd, fallback, output_json)


def call_predict_api(features: dict[str, Any]) -> dict:
    """予測APIサーバーにリクエストを送信する。"""
    print(f"\n[Pipeline] Step 4: 予測API呼び出し...")
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{PREDICTOR_URL}/predict",
                json=features,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  APIリクエスト失敗 ({attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(5)

    return {
        "recovery_probability": 0.5,
        "risk_level": "不明（API接続不可）",
        "recommended_action": "手動評価を実施してください",
        "shap_values": {},
        "top_5_factors": [],
    }


def generate_final_report(
    patient_id: str,
    clinical_data: dict,
    nnunet_result: dict,
    totalseg_result: dict,
    bianca_result: dict,
    prediction: dict,
    output_dir: str,
) -> dict:
    """全ツールの結果を統合して最終レポートを生成する。"""
    report = {
        "report_metadata": {
            "patient_id": patient_id,
            "generated_at": datetime.now().isoformat(),
            "pipeline_version": "1.0.0",
            "disclaimer": "本レポートは研究目的のAI出力です。臨床判断は医師が行ってください。",
        },
        "imaging_quantification": {
            "infarct": {
                "volume_ml": nnunet_result.get("lesion_volume_ml"),
                "severity":  nnunet_result.get("severity"),
                "hotspot_affected_count": nnunet_result.get("hotspot_affected_count"),
                "dysphagia_risk": nnunet_result.get("dysphagia_risk_from_lesion"),
            },
            "white_matter_hyperintensity": {
                "volume_ml":       bianca_result.get("wmh_volume_ml"),
                "fazekas_grade":   bianca_result.get("fazekas_grade"),
                "dysphagia_risk":  bianca_result.get("dysphagia_risk"),
                "wml_index":       bianca_result.get("wml_index"),
            },
            "swallowing_muscles": {
                "tongue_csa_cm2": totalseg_result.get("tongue_csa_cm2"),
                "spc_volume_cm3": totalseg_result.get("superior_pharyngeal_constrictor_volume_cm3"),
                "overall_muscle_risk": totalseg_result.get("overall_assessment", {}).get("overall_risk"),
            },
        },
        "prediction": {
            "recovery_probability": prediction.get("recovery_probability"),
            "risk_level":           prediction.get("risk_level"),
            "recommended_action":   prediction.get("recommended_action"),
            "top_5_shap_factors":   prediction.get("top_5_factors"),
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{patient_id}_final_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[Pipeline] 最終レポート保存: {report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def run_pipeline(
    patient_id: str,
    dicom_dir: str,
    output_base: str,
    clinical_data: dict,
) -> dict:
    """パイプライン全体を実行する。"""
    print(f"\n{'='*60}")
    print(f"嚥下障害AIパイプライン 開始: {patient_id}")
    print(f"{'='*60}")

    output_dir = os.path.join(output_base, patient_id)

    # Step 1: nnU-Net
    nnunet_result = run_nnunet(
        patient_id=patient_id,
        input_path=dicom_dir,
        output_dir=os.path.join(output_dir, "nnunet"),
        is_dicom=True,
    )

    # Step 2: TotalSegmentator
    nifti_path = os.path.join(output_dir, "nnunet", f"{patient_id}_0000.nii.gz")
    totalseg_result = run_totalseg(
        patient_id=patient_id,
        input_nifti=nifti_path,
        output_dir=os.path.join(output_dir, "totalseg"),
        sex=clinical_data.get("sex", "male"),
    )

    # Step 3: BIANCA
    flair_path = clinical_data.get("flair_path", nifti_path)
    bianca_result = run_bianca(
        patient_id=patient_id,
        flair_path=flair_path,
        output_dir=os.path.join(output_dir, "bianca"),
        t1_path=clinical_data.get("t1_path"),
        age=clinical_data.get("age", 65),
    )

    # Step 4: 予測API
    api_features = {
        "patient_id":    patient_id,
        "aspects_score": clinical_data.get("aspects_score", 7),
        "lesion_volume_ml": nnunet_result.get("lesion_volume_ml", 0),
        "wmh_volume_ml":    bianca_result.get("wmh_volume_ml", 0),
        "tongue_csa_cm2":   totalseg_result.get("tongue_csa_cm2", 4.0),
        "superior_pharyngeal_constrictor_volume_cm3": totalseg_result.get(
            "superior_pharyngeal_constrictor_volume_cm3", 2.0
        ),
        "hotspot_affected_count": nnunet_result.get("hotspot_affected_count", 0),
        "nihss_score":   clinical_data.get("nihss_score", 5),
        "age":           clinical_data.get("age", 65),
        "albumin_gdl":   clinical_data.get("albumin_gdl", 3.8),
        "bmi":           clinical_data.get("bmi", 22.0),
        "prior_stroke":  clinical_data.get("prior_stroke", 0),
        "atrial_fibrillation": clinical_data.get("atrial_fibrillation", 0),
        "diabetes":      clinical_data.get("diabetes", 0),
        "sex_male":      1 if clinical_data.get("sex", "male") == "male" else 0,
    }
    prediction = call_predict_api(api_features)

    return generate_final_report(
        patient_id=patient_id,
        clinical_data=clinical_data,
        nnunet_result=nnunet_result,
        totalseg_result=totalseg_result,
        bianca_result=bianca_result,
        prediction=prediction,
        output_dir=output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="嚥下障害AI統合パイプライン")
    parser.add_argument("--patient-id", required=True, help="患者ID")
    parser.add_argument("--dicom-dir", required=True, help="DICOMディレクトリ")
    parser.add_argument("--output", default="/data/output", help="出力ベースディレクトリ")
    parser.add_argument("--clinical-json", help="臨床データJSON（aspects_score, nihss等）")
    args = parser.parse_args()

    clinical_data = {}
    if args.clinical_json and os.path.exists(args.clinical_json):
        with open(args.clinical_json, encoding="utf-8") as f:
            clinical_data = json.load(f)

    run_pipeline(
        patient_id=args.patient_id,
        dicom_dir=args.dicom_dir,
        output_base=args.output,
        clinical_data=clinical_data,
    )


if __name__ == "__main__":
    main()
