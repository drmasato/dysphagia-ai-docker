"""
nnU-Net v2 推論スクリプト
DICOM/NIFTI入力から梗塞巣を自動セグメンテーションし、体積・部位を定量化する。
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk


SEVERITY_THRESHOLDS = {
    "small":    (0,   5),
    "moderate": (5,   30),
    "large":    (30,  100),
    "massive":  (100, float("inf")),
}

SEVERITY_DYSPHAGIA_RISK = {
    "small":    "低リスク",
    "moderate": "中リスク",
    "large":    "高リスク",
    "massive":  "最高リスク",
}

# 嚥下ホットスポット（ASPECTS領域）
SWALLOWING_HOTSPOTS = ["M5", "M6", "IC", "MCA_DEEP"]


def dicom_to_nifti(dicom_dir: str, output_path: str) -> str:
    """DICOMシリーズをNIFTI形式に変換する。"""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not dicom_names:
        raise ValueError(f"DICOMファイルが見つかりません: {dicom_dir}")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # HU値クリッピング（脳実質: -50〜150 HU）
    clamp = sitk.ClampImageFilter()
    clamp.SetLowerBound(-50)
    clamp.SetUpperBound(150)
    image = clamp.Execute(sitk.Cast(image, sitk.sitkFloat32))

    sitk.WriteImage(image, output_path)
    print(f"[nnU-Net] DICOM→NIFTI変換完了: {output_path}")
    return output_path


def run_nnunet_inference(
    input_nifti: str,
    output_dir: str,
    dataset_id: int = 500,
    configuration: str = "3d_fullres",
    folds: str = "all",
) -> str:
    """nnU-Netコマンドラインを呼び出して推論を実行する。"""
    os.makedirs(output_dir, exist_ok=True)
    input_dir = str(Path(input_nifti).parent)

    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", str(dataset_id),
        "-c", configuration,
        "-f", folds,
        "--save_probabilities",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nnU-Net推論エラー:\n{result.stderr}")

    output_files = list(Path(output_dir).glob("*.nii.gz"))
    if not output_files:
        raise FileNotFoundError(f"出力ファイルが見つかりません: {output_dir}")

    print(f"[nnU-Net] 推論完了: {output_files[0]}")
    return str(output_files[0])


def calculate_lesion_volume(
    segmentation_path: str,
    voxel_spacing_mm: tuple = None,
) -> dict:
    """セグメンテーションマスクから梗塞体積と重症度を算出する。"""
    seg = sitk.ReadImage(segmentation_path)
    seg_array = sitk.GetArrayFromImage(seg)

    if voxel_spacing_mm is None:
        spacing = seg.GetSpacing()
        voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    else:
        voxel_vol_mm3 = voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]

    lesion_voxels = int(np.sum(seg_array > 0))
    volume_ml = lesion_voxels * voxel_vol_mm3 / 1000.0

    severity = "small"
    for level, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= volume_ml < hi:
            severity = level
            break

    # 嚥下ホットスポット擬似評価（実装では部位別マスクが必要）
    hotspot_count = _estimate_hotspot_count(volume_ml)

    return {
        "lesion_volume_ml": round(volume_ml, 2),
        "lesion_voxels": lesion_voxels,
        "severity": severity,
        "dysphagia_risk_from_lesion": SEVERITY_DYSPHAGIA_RISK[severity],
        "hotspot_affected_count": hotspot_count,
        "segmentation_path": segmentation_path,
    }


def _estimate_hotspot_count(volume_ml: float) -> int:
    """梗塞体積から嚥下ホットスポット関与数を推定する（近似）。"""
    if volume_ml < 5:
        return 0
    elif volume_ml < 15:
        return 1
    elif volume_ml < 40:
        return 2
    else:
        return 3


def run_inference(
    input_path: str,
    output_dir: str,
    is_dicom: bool,
    patient_id: str,
    dataset_id: int = 500,
) -> dict:
    """エンドツーエンドの推論パイプラインを実行する。"""
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        if is_dicom:
            nifti_path = os.path.join(tmpdir, f"{patient_id}_0000.nii.gz")
            dicom_to_nifti(input_path, nifti_path)
        else:
            nifti_path = input_path

        seg_dir = os.path.join(output_dir, "segmentation")
        seg_path = run_nnunet_inference(nifti_path, seg_dir, dataset_id=dataset_id)
        result = calculate_lesion_volume(seg_path)

    result["patient_id"] = patient_id
    output_json = os.path.join(output_dir, f"{patient_id}_nnunet_result.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[nnU-Net] 結果保存: {output_json}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="nnU-Net v2 梗塞巣推論スクリプト")
    parser.add_argument("--input", required=True, help="入力パス（DICOMディレクトリ or NIFTIファイル）")
    parser.add_argument("--output", required=True, help="出力ディレクトリ")
    parser.add_argument("--dicom", action="store_true", help="入力がDICOMの場合に指定")
    parser.add_argument("--patient-id", default="unknown", help="患者ID")
    parser.add_argument("--dataset-id", type=int, default=500, help="nnU-Net Dataset ID")
    args = parser.parse_args()

    run_inference(
        input_path=args.input,
        output_dir=args.output,
        is_dicom=args.dicom,
        patient_id=args.patient_id,
        dataset_id=args.dataset_id,
    )


if __name__ == "__main__":
    main()
