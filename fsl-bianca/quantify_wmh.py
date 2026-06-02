"""
WMH体積定量化・Fazekas分類スクリプト
BIANCA出力マスクからFazekas分類・年齢補正WML指数を算出する。
"""

import argparse
import json
import os

import nibabel as nib
import numpy as np


FAZEKAS_THRESHOLDS = [
    (0,    2.0,  1, "点状（Grade 1）",       "低リスク",  0.05),
    (2.0,  10.0, 2, "始まりの融合（Grade 2）", "中リスク",  0.20),
    (10.0, float("inf"), 3, "高度融合（Grade 3）", "高リスク",  0.375),
]

DYSPHAGIA_RISK_DETAIL = {
    1: {"risk": "低リスク",  "silent_aspiration_rate": "<5%",    "action": "経過観察"},
    2: {"risk": "中リスク",  "silent_aspiration_rate": "15-25%", "action": "RSST・MWST実施"},
    3: {"risk": "高リスク",  "silent_aspiration_rate": "30-45%", "action": "VE/VF優先実施・ST緊急紹介"},
}


def classify_fazekas(wmh_volume_ml: float) -> dict:
    """WMH体積からFazekas分類を決定する。"""
    for lo, hi, grade, desc, risk, rate in FAZEKAS_THRESHOLDS:
        if lo <= wmh_volume_ml < hi:
            return {
                "fazekas_grade": grade,
                "fazekas_description": desc,
                "dysphagia_risk": risk,
                "silent_aspiration_rate": rate,
                **DYSPHAGIA_RISK_DETAIL[grade],
            }
    return {"fazekas_grade": 3, "fazekas_description": "高度融合（Grade 3）", "dysphagia_risk": "高リスク"}


def calculate_wml_index(wmh_volume_ml: float, age: int) -> dict:
    """年齢補正WML指数を算出する（Fazekas 2005 正規化式に基づく近似）。"""
    expected_volume = max(0.0, (age - 50) * 0.15)
    if expected_volume == 0:
        wml_index = float("inf") if wmh_volume_ml > 0 else 0.0
    else:
        wml_index = wmh_volume_ml / expected_volume

    if wml_index >= 3.0:
        interpretation = "年齢以上（高度過剰）"
    elif wml_index >= 2.0:
        interpretation = "年齢以上（中等度過剰）"
    elif wml_index >= 1.0:
        interpretation = "年齢相当"
    else:
        interpretation = "年齢以下"

    return {
        "expected_volume_ml": round(expected_volume, 2),
        "wml_index": round(wml_index, 2) if wml_index != float("inf") else 999.0,
        "wml_index_interpretation": interpretation,
    }


def quantify_from_mask(mask_path: str) -> float:
    """NIFTIマスクから直接WMH体積を計算する。"""
    img = nib.load(mask_path)
    data = img.get_fdata()
    zooms = img.header.get_zooms()
    voxel_vol_mm3 = zooms[0] * zooms[1] * zooms[2]
    voxel_count = int(np.sum(data > 0.5))
    return voxel_count * voxel_vol_mm3 / 1000.0


def run_quantification(
    wmh_mask_path: str,
    output_dir: str,
    patient_id: str,
    age: int = 65,
    wmh_volume_ml: float = None,
) -> dict:
    """WMHマスクから完全な定量評価レポートを生成する。"""
    os.makedirs(output_dir, exist_ok=True)

    if wmh_volume_ml is None:
        wmh_volume_ml = quantify_from_mask(wmh_mask_path)

    fazekas = classify_fazekas(wmh_volume_ml)
    wml = calculate_wml_index(wmh_volume_ml, age)

    result = {
        "patient_id": patient_id,
        "wmh_volume_ml": round(wmh_volume_ml, 3),
        **fazekas,
        **wml,
        "wmh_mask_path": wmh_mask_path,
    }

    output_json = os.path.join(output_dir, f"{patient_id}_wmh_result.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[BIANCA] WMH定量結果保存: {output_json}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="WMH定量化・Fazekas分類")
    parser.add_argument("--wmh-mask", required=True, help="WMHバイナリマスクNIFTIパス")
    parser.add_argument("--output-dir", required=True, help="出力ディレクトリ")
    parser.add_argument("--patient-id", default="unknown", help="患者ID")
    parser.add_argument("--age", type=int, default=65, help="患者年齢（年齢補正に使用）")
    parser.add_argument("--wmh-volume-ml", type=float, default=None, help="既知のWMH体積（mL）")
    args = parser.parse_args()

    run_quantification(
        wmh_mask_path=args.wmh_mask,
        output_dir=args.output_dir,
        patient_id=args.patient_id,
        age=args.age,
        wmh_volume_ml=args.wmh_volume_ml,
    )


if __name__ == "__main__":
    main()
