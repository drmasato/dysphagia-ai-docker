"""
TotalSegmentator 嚥下筋定量化スクリプト

タスク優先順位（auto モード）:
  1. headneck_muscles_highres  ライセンス必要・最高精度
  2. headneck_muscles           ライセンス必要・標準精度
  3. head_muscles               無料 (v2.3以上)・舌筋/咬筋/二腹筋を直接計測
     + head_glands_cavities     無料 (v2.3以上)・咽頭腔形態
  4. total                      無料・脳体積から嚥下筋を近似推定（最低精度）

ライセンス申請: https://backend.totalsegmentator.com/license-academic/
ライセンス設定: TotalSegmentator --license_number <18文字の番号>
              or: python3 -c "from totalsegmentator.config import set_license_number; set_license_number('<number>')"
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 筋肉定義
# ──────────────────────────────────────────────────────────────────────────────

# headneck_muscles タスクの構造（ライセンス必要）
LICENSED_MUSCLES = {
    "superior_pharyngeal_constrictor":  "上咽頭収縮筋",
    "middle_pharyngeal_constrictor":    "中咽頭収縮筋",
    "inferior_pharyngeal_constrictor":  "下咽頭収縮筋",
    "sternocleidomastoid_left":         "胸鎖乳突筋（左）",
    "sternocleidomastoid_right":        "胸鎖乳突筋（右）",
    "trapezius_left":                   "僧帽筋（左）",
    "trapezius_right":                  "僧帽筋（右）",
    "platysma_left":                    "広頸筋（左）",
    "platysma_right":                   "広頸筋（右）",
    "levator_scapulae_left":            "肩甲挙筋（左）",
    "levator_scapulae_right":           "肩甲挙筋（右）",
    "anterior_scalene_left":            "前斜角筋（左）",
    "anterior_scalene_right":           "前斜角筋（右）",
    "middle_scalene_left":              "中斜角筋（左）",
    "middle_scalene_right":             "中斜角筋（右）",
    "posterior_scalene_left":           "後斜角筋（左）",
    "posterior_scalene_right":          "後斜角筋（右）",
    "sterno_thyroid_left":              "胸骨甲状筋（左）",
    "sterno_thyroid_right":             "胸骨甲状筋（右）",
    "thyrohyoid_left":                  "甲状舌骨筋（左）",
    "thyrohyoid_right":                 "甲状舌骨筋（右）",
    "prevertebral_left":                "椎前筋（左）",
    "prevertebral_right":               "椎前筋（右）",
}

# head_muscles タスクの構造（無料・v2.3以上）
FREE_MUSCLES = {
    "tongue":               "舌筋",
    "masseter_left":        "咬筋（左）",
    "masseter_right":       "咬筋（右）",
    "temporalis_left":      "側頭筋（左）",
    "temporalis_right":     "側頭筋（右）",
    "medial_pterygoid_left":  "内側翼突筋（左）",
    "medial_pterygoid_right": "内側翼突筋（右）",
    "lateral_pterygoid_left": "外側翼突筋（左）",
    "lateral_pterygoid_right":"外側翼突筋（右）",
    "digastric_left":       "二腹筋（左）",
    "digastric_right":      "二腹筋（右）",
}

ALL_MUSCLES = {**FREE_MUSCLES, **LICENSED_MUSCLES}

# サルコペニア閾値（Ogawa et al. 2018）
SARCOPENIA_THRESHOLDS = {
    "tongue": {
        "male":   {"low": 4.0, "severe": 3.0},
        "female": {"low": 3.5, "severe": 2.5},
    },
    "superior_pharyngeal_constrictor": {
        "male":   {"low": 1.5, "severe": 1.0},
        "female": {"low": 1.2, "severe": 0.8},
    },
    "masseter_left": {
        "male":   {"low": 5.0, "severe": 3.5},
        "female": {"low": 4.0, "severe": 3.0},
    },
    "masseter_right": {
        "male":   {"low": 5.0, "severe": 3.5},
        "female": {"low": 4.0, "severe": 3.0},
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# TotalSegmentator 実行
# ──────────────────────────────────────────────────────────────────────────────

def run_totalsegmentator(input_nifti: str, output_dir: str, task: str) -> bool:
    """指定タスクでTotalSegmentatorを実行する。失敗時はFalseを返す。"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["TotalSegmentator", "-i", input_nifti, "-o", output_dir, "--task", task]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = result.stderr[-300:] if result.stderr else result.stdout[-300:]
        print(f"[TotalSeg] {task} 失敗: {err}")
        return False
    print(f"[TotalSeg] セグメンテーション完了 (task={task})")
    return True


def assess_ct_coverage(input_nifti: str) -> dict:
    """
    CT画像の撮影範囲を評価する。
    嚥下筋計測には頭部（脳）+ 頸部（C1-C7）の coverage が必要。
    """
    import SimpleITK as sitk
    img = sitk.ReadImage(input_nifti)
    size = img.GetSize()          # (x, y, z) in voxels
    spacing = img.GetSpacing()    # (dx, dy, dz) in mm

    fov_z_mm = size[2] * spacing[2]  # Z方向の撮影範囲 mm

    # 判定基準:
    # - 脳CT（stroke用）: 通常 130-160mm  → 口腔・頸部は含まない
    # - head+neck CT:     通常 300-400mm  → 舌・咽頭筋を含む
    coverage = "brain_only"
    if fov_z_mm >= 250:
        coverage = "head_and_neck"
    elif fov_z_mm >= 180:
        coverage = "head_partial_neck"

    return {
        "fov_z_mm": round(fov_z_mm, 1),
        "n_slices": size[2],
        "slice_thickness_mm": round(spacing[2], 3),
        "coverage": coverage,
        "coverage_warning": (
            None if coverage == "head_and_neck" else
            f"撮影範囲 {fov_z_mm:.0f}mm は脳のみをカバーしています。"
            "舌筋・咽頭収縮筋の計測には頭頸部CT（≥250mm）が必要です。"
        ),
    }


def select_task_auto(input_nifti: str, seg_dir: str) -> tuple[str, bool, list[str]]:
    """
    利用可能な最高精度タスクを自動選択して実行する。
    戻り値: (主タスク名, 推定フラグ, 実行タスクリスト)
    """
    executed_tasks = []

    # 優先順位1: headneck_muscles_highres（ライセンス）
    if run_totalsegmentator(input_nifti, seg_dir, "headneck_muscles_highres"):
        executed_tasks.append("headneck_muscles_highres")
        # 舌筋・咬筋を追加計測（head_muscles）
        if run_totalsegmentator(input_nifti, seg_dir + "_head", "head_muscles"):
            executed_tasks.append("head_muscles")
            _merge_seg_dirs(seg_dir + "_head", seg_dir)
        return "headneck_muscles_highres", False, executed_tasks

    # 優先順位2: headneck_muscles（ライセンス）
    if run_totalsegmentator(input_nifti, seg_dir, "headneck_muscles"):
        executed_tasks.append("headneck_muscles")
        # 舌筋・咬筋を追加（head_muscles）
        if run_totalsegmentator(input_nifti, seg_dir + "_head", "head_muscles"):
            executed_tasks.append("head_muscles")
            _merge_seg_dirs(seg_dir + "_head", seg_dir)
        return "headneck_muscles+head_muscles", False, executed_tasks

    # 優先順位3: head_muscles（無料・v2.3以上）
    print("[TotalSeg] ライセンスなし。head_muscles（無料）で舌筋・咬筋を直接計測します...")
    if run_totalsegmentator(input_nifti, seg_dir, "head_muscles"):
        executed_tasks.append("head_muscles")
        return "head_muscles", False, executed_tasks

    # 優先順位4: total（無料・推定）
    print("[TotalSeg] head_muscles使用不可。total タスクで脳体積から近似推定します...")
    if run_totalsegmentator(input_nifti, seg_dir, "total"):
        executed_tasks.append("total")
        return "total", True, executed_tasks

    raise RuntimeError("すべてのTotalSegmentatorタスクが失敗しました")


def _merge_seg_dirs(src_dir: str, dst_dir: str):
    """srcのマスクファイルをdstにコピー（上書きなし）。"""
    import shutil
    if not os.path.exists(src_dir):
        return
    for f in os.listdir(src_dir):
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f)
        if f.endswith(".nii.gz") and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)


# ──────────────────────────────────────────────────────────────────────────────
# 定量化
# ──────────────────────────────────────────────────────────────────────────────

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
        "volume_cm3":   round(volume_cm3, 3),
        "max_csa_cm2":  round(max_csa_cm2, 3),
        "mean_csa_cm2": round(mean_csa_cm2, 3),
        "voxel_count":  voxel_count,
        "estimated":    False,
    }


def estimate_muscles_from_brain_volume(brain_vol_cm3: float, sex: str) -> dict:
    """脳体積から嚥下筋体積を近似推定する（totalタスク用フォールバック）。"""
    scale = (brain_vol_cm3 / 1200.0) if brain_vol_cm3 > 100 else 1.0
    ref = {
        "tongue":               (40.0 if sex == "male" else 34.0, 4.2 if sex == "male" else 3.7),
        "superior_pharyngeal_constrictor": (5.0 if sex == "male" else 4.0, 1.6 if sex == "male" else 1.3),
        "middle_pharyngeal_constrictor":   (5.5 if sex == "male" else 4.5, 1.2 if sex == "male" else 1.0),
        "inferior_pharyngeal_constrictor": (6.0 if sex == "male" else 5.0, 1.0 if sex == "male" else 0.8),
        "sternocleidomastoid_left":        (60.0 if sex == "male" else 45.0, 6.5 if sex == "male" else 5.0),
        "sternocleidomastoid_right":       (60.0 if sex == "male" else 45.0, 6.5 if sex == "male" else 5.0),
        "masseter_left":                   (30.0 if sex == "male" else 22.0, 5.5 if sex == "male" else 4.2),
        "masseter_right":                  (30.0 if sex == "male" else 22.0, 5.5 if sex == "male" else 4.2),
        "digastric_left":                  (10.0, 1.5),
        "digastric_right":                 (10.0, 1.5),
    }
    note = "total タスクの脳体積から近似推定。head_muscles(無料)またはheadneck_muscles(学術ライセンス)で実測可"
    results = {}
    for muscle_key in ALL_MUSCLES:
        vol_ref, csa_ref = ref.get(muscle_key, (10.0, 1.5))
        results[muscle_key] = {
            "volume_cm3":   round(vol_ref * scale, 3),
            "max_csa_cm2":  round(csa_ref * scale, 3),
            "mean_csa_cm2": round(csa_ref * scale * 0.7, 3),
            "voxel_count":  0,
            "estimated":    True,
            "note":         note,
        }
    return results


def evaluate_sarcopenia(muscle_key: str, csa_cm2: float, sex: str) -> dict:
    if muscle_key not in SARCOPENIA_THRESHOLDS:
        return {"sarcopenia_grade": "N/A", "below_threshold": False}
    thr = SARCOPENIA_THRESHOLDS[muscle_key][sex]
    if csa_cm2 < thr["severe"]:
        grade, below = "重度低値", True
    elif csa_cm2 < thr["low"]:
        grade, below = "低値", True
    else:
        grade, below = "正常", False
    return {"sarcopenia_grade": grade, "below_threshold": below,
            "threshold_low": thr["low"], "threshold_severe": thr["severe"]}


def calculate_overall_dysphagia_risk(muscles_data: dict) -> dict:
    high  = sum(1 for d in muscles_data.values() if d.get("sarcopenia_grade") == "重度低値")
    mod   = sum(1 for d in muscles_data.values() if d.get("sarcopenia_grade") == "低値")
    if high >= 2 or (high >= 1 and mod >= 2):
        risk, rec = "高リスク", "VE/VF検査の優先実施・ST緊急紹介"
    elif high >= 1 or mod >= 3:
        risk, rec = "中リスク", "RSST・MWST実施後にST紹介"
    else:
        risk, rec = "低リスク", "経過観察・食事中の観察継続"
    return {"overall_risk": risk, "recommendation": rec,
            "high_risk_muscles": high, "moderate_risk_muscles": mod}


# ──────────────────────────────────────────────────────────────────────────────
# メインパイプライン
# ──────────────────────────────────────────────────────────────────────────────

def run_quantification(
    input_nifti: str,
    output_dir: str,
    patient_id: str,
    sex: str = "male",
    task: str = "auto",
    skip_segmentation: bool = False,
    license_number: str = None,
) -> dict:
    """エンドツーエンドの嚥下筋定量化パイプラインを実行する。"""
    os.makedirs(output_dir, exist_ok=True)
    seg_dir = os.path.join(output_dir, "segmentation")

    # ライセンス設定（引数で渡された場合）
    if license_number:
        _set_license(license_number)

    estimated_all = False
    use_task = task

    # CT撮影範囲評価
    ct_coverage = assess_ct_coverage(input_nifti)
    print(f"[TotalSeg] CT撮影範囲: {ct_coverage['fov_z_mm']}mm ({ct_coverage['coverage']})")
    if ct_coverage["coverage_warning"]:
        print(f"[TotalSeg] ⚠ {ct_coverage['coverage_warning']}")

    executed_tasks = [task]
    if not skip_segmentation:
        if task == "auto":
            use_task, estimated_all, executed_tasks = select_task_auto(input_nifti, seg_dir)
        else:
            ok = run_totalsegmentator(input_nifti, seg_dir, task)
            if not ok:
                raise RuntimeError(f"タスク '{task}' の実行に失敗しました")
            estimated_all = (task == "total")

    # マスクから直接計測 or 推定
    muscles_data: dict = {}

    if estimated_all and os.path.exists(os.path.join(seg_dir, "brain.nii.gz")):
        # total タスク: 脳体積から推定
        brain_img = nib.load(os.path.join(seg_dir, "brain.nii.gz"))
        brain_data = brain_img.get_fdata()
        zooms = brain_img.header.get_zooms()
        brain_vol = np.sum(brain_data > 0.5) * zooms[0] * zooms[1] * zooms[2] / 1000.0
        print(f"[TotalSeg] 脳体積: {brain_vol:.0f} cm³")
        muscles_data = estimate_muscles_from_brain_volume(brain_vol, sex)

    # 実マスクが存在する構造は直接計測値で上書き
    for muscle_key, muscle_name in ALL_MUSCLES.items():
        mask_path = os.path.join(seg_dir, f"{muscle_key}.nii.gz")
        if os.path.exists(mask_path):
            metrics = quantify_mask(mask_path)
            sarco = evaluate_sarcopenia(muscle_key, metrics["max_csa_cm2"], sex)
            muscles_data[muscle_key] = {"name_ja": muscle_name, **metrics, **sarco}
        elif muscle_key not in muscles_data:
            muscles_data[muscle_key] = {
                "name_ja": muscle_name,
                "volume_cm3": 0.0, "max_csa_cm2": 0.0, "mean_csa_cm2": 0.0,
                "voxel_count": 0, "estimated": False,
                "sarcopenia_grade": "N/A", "below_threshold": False,
            }
        # name_ja と sarcopenia が欠けている場合に補完
        muscles_data[muscle_key].setdefault("name_ja", muscle_name)
        if "sarcopenia_grade" not in muscles_data[muscle_key]:
            sarco = evaluate_sarcopenia(muscle_key, muscles_data[muscle_key].get("max_csa_cm2", 0), sex)
            muscles_data[muscle_key].update(sarco)

    # 計測タスク種別をサマリに付記
    for v in muscles_data.values():
        has_real_voxels = v.get("voxel_count", 0) > 0
        if not v.get("estimated", True) and has_real_voxels:
            v["measurement_type"] = "直接計測"
        elif not v.get("estimated", True) and not has_real_voxels:
            v["measurement_type"] = "範囲外（CTカバレッジ不足）"
        else:
            v["measurement_type"] = "近似推定"

    overall = calculate_overall_dysphagia_risk(muscles_data)

    # 直接計測できた構造数（voxelが実際に存在するもののみ）
    directly_measured = sum(
        1 for v in muscles_data.values()
        if not v.get("estimated", True) and v.get("voxel_count", 0) > 0
    )

    result = {
        "patient_id": patient_id,
        "sex": sex,
        "task_used": use_task,
        "executed_tasks": executed_tasks,
        "ct_coverage": ct_coverage,
        "directly_measured_structures": directly_measured,
        "total_structures": len(muscles_data),
        "muscles": muscles_data,
        "overall_assessment": overall,
        # 予測APIが参照するトップレベルキー
        "tongue_csa_cm2": muscles_data.get("tongue", {}).get("max_csa_cm2", 0.0),
        "superior_pharyngeal_constrictor_volume_cm3": muscles_data.get(
            "superior_pharyngeal_constrictor", {}
        ).get("volume_cm3", 0.0),
        "masseter_left_csa_cm2": muscles_data.get("masseter_left", {}).get("max_csa_cm2", 0.0),
        "measurement_quality": _assess_measurement_quality(use_task, directly_measured),
    }

    output_json = os.path.join(output_dir, f"{patient_id}_muscles_result.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    _print_summary(result)
    return result


def _set_license(license_number: str):
    """TotalSegmentatorライセンスを設定する。"""
    try:
        from totalsegmentator.config import set_license_number
        set_license_number(license_number, skip_validation=False)
        print(f"[TotalSeg] ライセンス設定完了")
    except Exception as e:
        print(f"[TotalSeg] ライセンス設定エラー: {e}")


def _assess_measurement_quality(task: str, measured_count: int) -> str:
    if "highres" in task:
        return "最高精度（headneck_muscles_highres: 直接計測）"
    elif "headneck_muscles" in task:
        return "高精度（headneck_muscles: 直接計測）"
    elif "head_muscles" in task:
        return "中精度（head_muscles: 舌筋・咬筋・二腹筋を直接計測）"
    elif task == "total" and measured_count == 0:
        return "低精度（total: 脳体積から近似推定。head_muscles(無料)またはheadneck_muscles(学術ライセンス)推奨）"
    return "不明"


def _print_summary(result: dict):
    print(f"[TotalSeg] 結果保存完了")
    cov = result.get("ct_coverage", {})
    summary = {
        "patient_id": result["patient_id"],
        "task_used": result["task_used"],
        "measurement_quality": result["measurement_quality"],
        "directly_measured": f"{result['directly_measured_structures']}/{result['total_structures']}",
        "ct_coverage": f"{cov.get('fov_z_mm','?')}mm ({cov.get('coverage','?')})",
        "tongue_csa_cm2": result["tongue_csa_cm2"],
        "masseter_left_csa_cm2": result["masseter_left_csa_cm2"],
        "overall_risk": result["overall_assessment"]["overall_risk"],
    }
    if cov.get("coverage_warning"):
        summary["coverage_warning"] = cov["coverage_warning"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# ライセンス申請ガイド出力
# ──────────────────────────────────────────────────────────────────────────────

def print_license_guide():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  TotalSegmentator headneck_muscles ライセンス取得ガイド           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  【学術・非商用ライセンス】無料で取得可能                          ║
║                                                                  ║
║  Step 1: 申請フォームにアクセス                                   ║
║    URL: https://backend.totalsegmentator.com/license-academic/   ║
║                                                                  ║
║  Step 2: フォームに記入（英語）                                    ║
║    - Name / Institution / Email                                  ║
║    - Intended use（例: "Dysphagia risk assessment in stroke"）    ║
║    - Non-commercial research use の確認                          ║
║                                                                  ║
║  Step 3: ライセンス番号を受け取る（18文字）                        ║
║                                                                  ║
║  Step 4: コンテナ内でライセンスを設定                             ║
║    docker exec dysphagia-totalseg                                ║
║      python3 /workspace/quantify_muscles.py --set-license <番号> ║
║                                                                  ║
║  Step 5: 動作確認                                                 ║
║    --task headneck_muscles                                       ║
║                                                                  ║
║  【直接計測できる構造】（ライセンス取得後）                        ║
║    上咽頭収縮筋・中咽頭収縮筋・下咽頭収縮筋                       ║
║    胸鎖乳突筋（左右）・僧帽筋・広頸筋 等23構造                    ║
║                                                                  ║
║  【現在無料で直接計測可能】head_muscles タスク                    ║
║    舌筋・咬筋・側頭筋・翼突筋・二腹筋 計11構造                    ║
║                                                                  ║
║  【商用ライセンス】                                               ║
║    Contact: jakob.wasserthal@usb.ch                              ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TotalSegmentator 嚥下筋定量化（head_muscles / headneck_muscles 対応）"
    )
    parser.add_argument("--input",         help="入力NIFTIファイル")
    parser.add_argument("--output",        help="出力ディレクトリ")
    parser.add_argument("--patient-id",    default="unknown")
    parser.add_argument("--sex",           choices=["male", "female"], default="male")
    parser.add_argument("--task",          default="auto",
        help="auto / head_muscles / headneck_muscles / headneck_muscles_highres / total")
    parser.add_argument("--skip-segmentation", action="store_true")
    parser.add_argument("--license-number", default=None,
        help="TotalSegmentatorライセンス番号（18文字）。headneck_musclsタスクに必要")
    parser.add_argument("--set-license",   default=None,
        help="ライセンス番号を設定して終了（セットアップ用）")
    parser.add_argument("--license-guide", action="store_true",
        help="ライセンス申請ガイドを表示")
    args = parser.parse_args()

    if args.license_guide:
        print_license_guide()
        return

    if args.set_license:
        _set_license(args.set_license)
        print("設定完了。次回から --task headneck_muscles が使用できます。")
        return

    if not args.input or not args.output:
        parser.error("--input と --output は必須です（--license-guide または --set-license を使う場合を除く）")

    run_quantification(
        input_nifti=args.input,
        output_dir=args.output,
        patient_id=args.patient_id,
        sex=args.sex,
        task=args.task,
        skip_segmentation=args.skip_segmentation,
        license_number=args.license_number,
    )


if __name__ == "__main__":
    main()
