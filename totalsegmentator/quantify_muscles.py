"""
TotalSegmentator 嚥下筋定量化スクリプト

TotalSegmentator v2 では headneck_muscles タスクにライセンスが必要。
- ライセンスあり  : --task headneck_muscles_highres で詳細定量化
- ライセンスなし  : --task total で利用可能な構造 (brain, skull) を使い
                   体積比から嚥下筋を近似推定（デモ用）
- --skip-segmentation: 既存マスクディレクトリを指定して直接定量化
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


SWALLOWING_MUSCLES = {
    "tongue":                          "舌筋",
    "superior_pharyngeal_constrictor": "上咽頭収縮筋",
    "middle_pharyngeal_constrictor":   "中咽頭収縮筋",
    "inferior_pharyngeal_constrictor": "下咽頭収縮筋",
    "sternocleidomastoid_left":        "胸鎖乳突筋（左）",
    "sternocleidomastoid_right":       "胸鎖乳突筋（右）",
    "masseter_left":                   "咬筋（左）",
    "masseter_right":                  "咬筋（右）",
    "digastric_left":                  "二腹筋（左）",
    "digastric_right":                 "二腹筋（右）",
}

SARCOPENIA_THRESHOLDS = {
    "tongue": {
        "male":   {"low": 4.0, "severe": 3.0},
        "female": {"low": 3.5, "severe": 2.5},
    },
    "superior_pharyngeal_constrictor": {
        "male":   {"low": 1.5, "severe": 1.0},
        "female": {"low": 1.2, "severe": 0.8},
    },
}

# フリータスクで取得できる頭部構造
FREE_TASK_HEAD_STRUCTURES = {
    "brain": "脳",
    "skull": "頭蓋骨",
}


def run_totalsegmentator(input_nifti: str, output_dir: str, task: str = "total") -> str:
    """TotalSegmentatorを実行する。"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "TotalSegmentator",
        "-i", input_nifti,
        "-o", output_dir,
        "--task", task,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"TotalSegmentator エラー（タスク={task}）:\n{result.stderr}")
    print(f"[TotalSeg] セグメンテーション完了 (task={task}): {output_dir}")
    return output_dir


def quantify_mask(mask_path: str) -> dict:
    """単一マスクから体積・断面積を算出する。"""
    img = nib.load(mask_path)
    data = img.get_fdata()
    zooms = img.header.get_zooms()
    voxel_vol_mm3 = zooms[0] * zooms[1] * zooms[2]
    voxel_area_mm2 = zooms[0] * zooms[1]
    voxel_count = int(np.sum(data > 0.5))
    volume_cm3 = voxel_count * voxel_vol_mm3 / 1000.0

    slice_areas = [
        float(np.sum(data[:, :, z] > 0.5)) * voxel_area_mm2
        for z in range(data.shape[2])
        if np.sum(data[:, :, z] > 0.5) > 0
    ]
    max_csa_cm2  = max(slice_areas) / 100.0 if slice_areas else 0.0
    mean_csa_cm2 = float(np.mean(slice_areas)) / 100.0 if slice_areas else 0.0

    return {
        "volume_cm3":    round(volume_cm3, 3),
        "max_csa_cm2":   round(max_csa_cm2, 3),
        "mean_csa_cm2":  round(mean_csa_cm2, 3),
        "voxel_count":   voxel_count,
    }


def estimate_muscles_from_total(seg_dir: str, sex: str) -> dict:
    """
    total タスクの脳・頭蓋骨体積から嚥下筋体積を近似推定する（デモ用）。
    推定式は健常成人の文献値比に基づく近似。
    """
    brain_path = os.path.join(seg_dir, "brain.nii.gz")
    skull_path  = os.path.join(seg_dir, "skull.nii.gz")

    brain_vol = 0.0
    if os.path.exists(brain_path):
        m = quantify_mask(brain_path)
        brain_vol = m["volume_cm3"]

    # 脳体積から筋肉体積を推定（健常者の脳1200mLを基準）
    scale = (brain_vol / 1200.0) if brain_vol > 100 else 1.0
    base_tongue = 40.0 if sex == "male" else 34.0
    base_spc    =  5.0 if sex == "male" else  4.0

    results = {}
    for muscle_key in SWALLOWING_MUSCLES:
        if "tongue" in muscle_key:
            est_vol = base_tongue * scale
            est_csa = (4.2 if sex == "male" else 3.7) * scale
        elif "superior_pharyngeal" in muscle_key:
            est_vol = base_spc * scale
            est_csa = (1.6 if sex == "male" else 1.3) * scale
        elif "sternocleidomastoid" in muscle_key:
            est_vol = 60.0 * scale
            est_csa = 6.5 * scale
        elif "masseter" in muscle_key:
            est_vol = 30.0 * scale
            est_csa = 5.0 * scale
        else:
            est_vol = 10.0 * scale
            est_csa = 1.5 * scale

        results[muscle_key] = {
            "name_ja": SWALLOWING_MUSCLES[muscle_key],
            "volume_cm3":   round(est_vol, 3),
            "max_csa_cm2":  round(est_csa, 3),
            "mean_csa_cm2": round(est_csa * 0.7, 3),
            "voxel_count":  0,
            "estimated":    True,
            "note": "total タスクの脳体積から近似推定（headneck_muscles ライセンスで実測可）",
        }

    return results


def evaluate_sarcopenia(muscle_key: str, csa_cm2: float, sex: str) -> dict:
    if muscle_key not in SARCOPENIA_THRESHOLDS:
        return {"sarcopenia_grade": "N/A", "below_threshold": False}
    thresholds = SARCOPENIA_THRESHOLDS[muscle_key][sex]
    if csa_cm2 < thresholds["severe"]:
        grade, below = "重度低値", True
    elif csa_cm2 < thresholds["low"]:
        grade, below = "低値", True
    else:
        grade, below = "正常", False
    return {
        "sarcopenia_grade": grade,
        "below_threshold": below,
        "threshold_low": thresholds["low"],
        "threshold_severe": thresholds["severe"],
    }


def calculate_overall_dysphagia_risk(muscles_data: dict) -> dict:
    high_count = sum(1 for d in muscles_data.values() if d.get("sarcopenia_grade") == "重度低値")
    mod_count  = sum(1 for d in muscles_data.values() if d.get("sarcopenia_grade") == "低値")

    if high_count >= 2 or (high_count >= 1 and mod_count >= 2):
        risk, rec = "高リスク", "VE/VF検査の優先実施・ST緊急紹介"
    elif high_count >= 1 or mod_count >= 3:
        risk, rec = "中リスク", "RSST・MWST実施後にST紹介"
    else:
        risk, rec = "低リスク", "経過観察・食事中の観察継続"

    return {
        "overall_risk": risk,
        "recommendation": rec,
        "high_risk_muscles": high_count,
        "moderate_risk_muscles": mod_count,
    }


def run_quantification(
    input_nifti: str,
    output_dir: str,
    patient_id: str,
    sex: str = "male",
    skip_segmentation: bool = False,
    task: str = "auto",
) -> dict:
    """エンドツーエンドの筋肉定量化パイプラインを実行する。"""
    os.makedirs(output_dir, exist_ok=True)
    seg_dir = os.path.join(output_dir, "segmentation")
    estimated = False

    if not skip_segmentation:
        # タスク自動選択: headneck_muscles があれば使い、なければ total にフォールバック
        use_task = task
        if task == "auto":
            try:
                run_totalsegmentator(input_nifti, seg_dir, "headneck_muscles_highres")
                use_task = "headneck_muscles_highres"
            except RuntimeError:
                try:
                    run_totalsegmentator(input_nifti, seg_dir, "headneck_muscles")
                    use_task = "headneck_muscles"
                except RuntimeError:
                    print("[TotalSeg] headneck_muscles ライセンス未取得。total タスクで近似推定します。")
                    run_totalsegmentator(input_nifti, seg_dir, "total")
                    use_task = "total"
        else:
            run_totalsegmentator(input_nifti, seg_dir, use_task)

        if use_task in ("total",):
            muscles_data = estimate_muscles_from_total(seg_dir, sex)
            estimated = True
        else:
            muscles_data = {}
    else:
        muscles_data = {}
        use_task = "manual"

    # 実マスクが存在する場合は上書き
    for muscle_key in SWALLOWING_MUSCLES:
        mask_path = os.path.join(seg_dir, f"{muscle_key}.nii.gz")
        if os.path.exists(mask_path):
            metrics = quantify_mask(mask_path)
            sarco   = evaluate_sarcopenia(muscle_key, metrics["max_csa_cm2"], sex)
            muscles_data[muscle_key] = {
                "name_ja": SWALLOWING_MUSCLES[muscle_key],
                **metrics,
                **sarco,
                "estimated": False,
            }
        elif muscle_key not in muscles_data:
            muscles_data[muscle_key] = {
                "name_ja": SWALLOWING_MUSCLES[muscle_key],
                "volume_cm3": 0.0, "max_csa_cm2": 0.0,
                "mean_csa_cm2": 0.0, "voxel_count": 0,
                "sarcopenia_grade": "N/A", "below_threshold": False,
                "estimated": False,
            }
        # サルコペニア評価が未済の場合は追加
        if "sarcopenia_grade" not in muscles_data.get(muscle_key, {}):
            sarco = evaluate_sarcopenia(muscle_key, muscles_data[muscle_key].get("max_csa_cm2", 0), sex)
            muscles_data[muscle_key].update(sarco)

    overall = calculate_overall_dysphagia_risk(muscles_data)

    result = {
        "patient_id": patient_id,
        "sex": sex,
        "task_used": use_task,
        "estimated": estimated,
        "muscles": muscles_data,
        "overall_assessment": overall,
        "tongue_csa_cm2": muscles_data.get("tongue", {}).get("max_csa_cm2", 0.0),
        "superior_pharyngeal_constrictor_volume_cm3": muscles_data.get(
            "superior_pharyngeal_constrictor", {}
        ).get("volume_cm3", 0.0),
    }

    output_json = os.path.join(output_dir, f"{patient_id}_muscles_result.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[TotalSeg] 結果保存: {output_json}")
    print(json.dumps({
        "patient_id": result["patient_id"],
        "task_used": result["task_used"],
        "estimated": result["estimated"],
        "tongue_csa_cm2": result["tongue_csa_cm2"],
        "overall_risk": result["overall_assessment"]["overall_risk"],
    }, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="TotalSegmentator 嚥下筋定量化")
    parser.add_argument("--input", required=True, help="入力NIFTIファイル")
    parser.add_argument("--output", required=True, help="出力ディレクトリ")
    parser.add_argument("--patient-id", default="unknown")
    parser.add_argument("--sex", choices=["male", "female"], default="male")
    parser.add_argument("--task", default="auto",
        help="TotalSegmentatorタスク (auto/total/headneck_muscles/headneck_muscles_highres)")
    parser.add_argument("--skip-segmentation", action="store_true")
    args = parser.parse_args()

    run_quantification(
        input_nifti=args.input,
        output_dir=args.output,
        patient_id=args.patient_id,
        sex=args.sex,
        skip_segmentation=args.skip_segmentation,
        task=args.task,
    )


if __name__ == "__main__":
    main()
