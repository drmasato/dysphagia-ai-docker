"""
DeepISLES 梗塞巣セグメンテーション統合スクリプト。
Nature Communications 2025 / ISLES'22 チャレンジ優勝アンサンブルモデル。

入力: DWI + ADC + FLAIR MRI NIfTI（nnU-Netではなく専用Docker使用）
出力: 梗塞巣バイナリマスク + 体積レポートJSON

Docker: isleschallenge/deepisles
事前準備: docker pull isleschallenge/deepisles

使用方法（直接実行）:
  python3 run_deepisles.py \
    --dwi  /path/dwi.nii.gz \
    --adc  /path/adc.nii.gz \
    --flair /path/flair.nii.gz \
    --output /output/PT-001 \
    --patient-id PT-001

  または DICOM ディレクトリ:
    --dwi-dicom  /dicom/dwi/ \
    --adc-dicom  /dicom/adc/ \
    --flair-dicom /dicom/flair/
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

DEEPISLES_IMAGE = "isleschallenge/deepisles"
DEEPISLES_SEVERITY = [
    (0,    5,   "極小梗塞",  "低リスク"),
    (5,    30,  "小梗塞",    "中リスク"),
    (30,   100, "中等度梗塞","高リスク"),
    (100,  float("inf"), "大梗塞", "最高リスク"),
]


def dicom_to_nifti(dicom_dir: str, output_path: str) -> str:
    """DICOMシリーズをNIfTI変換する（SimpleITK使用）。"""
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not ids:
        raise ValueError(f"DICOMが見つかりません: {dicom_dir}")
    names = reader.GetGDCMSeriesFileNames(dicom_dir, ids[0])
    reader.SetFileNames(names)
    img = reader.Execute()
    sitk.WriteImage(img, output_path)
    return output_path


def run_deepisles(
    dwi_path: str,
    adc_path: str,
    flair_path: str,
    output_dir: str,
    skull_strip: bool = True,
    fast: bool = False,
) -> str:
    """DeepISLES Dockerコンテナを実行して梗塞巣マスクを生成する。"""
    os.makedirs(output_dir, exist_ok=True)

    # DeepISLES は /app/data をデータルートとしてマウント
    # 入力ファイルを一時ディレクトリにまとめる
    data_dir = output_dir
    dwi_name   = "dwi.nii.gz"
    adc_name   = "adc.nii.gz"
    flair_name = "flair.nii.gz"

    import shutil
    if os.path.abspath(dwi_path)   != os.path.abspath(os.path.join(data_dir, dwi_name)):
        shutil.copy(dwi_path,   os.path.join(data_dir, dwi_name))
    if os.path.abspath(adc_path)   != os.path.abspath(os.path.join(data_dir, adc_name)):
        shutil.copy(adc_path,   os.path.join(data_dir, adc_name))
    if os.path.abspath(flair_path) != os.path.abspath(os.path.join(data_dir, flair_name)):
        shutil.copy(flair_path, os.path.join(data_dir, flair_name))

    cmd = [
        "docker", "run", "--rm",
        "--runtime=nvidia",
        "-v", f"{os.path.abspath(data_dir)}:/app/data",
        DEEPISLES_IMAGE,
        "--dwi_file_name",   dwi_name,
        "--adc_file_name",   adc_name,
        "--flair_file_name", flair_name,
    ]
    if skull_strip:
        cmd += ["--skull_strip", "True"]
    if fast:
        cmd += ["--fast", "True"]

    print(f"[DeepISLES] 実行中... (skull_strip={skull_strip}, fast={fast})")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"DeepISLES エラー:\n{result.stderr[-500:]}")

    # 出力: results/ensemble_prediction.nii.gz
    mask_path = os.path.join(data_dir, "results", "ensemble_prediction.nii.gz")
    if not os.path.exists(mask_path):
        # 出力ファイルを探す
        for root, _, files in os.walk(data_dir):
            for f in files:
                if "prediction" in f.lower() and f.endswith(".nii.gz"):
                    mask_path = os.path.join(root, f)
                    break

    print(f"[DeepISLES] マスク生成: {mask_path}")
    return mask_path


def quantify_infarct(mask_path: str) -> dict:
    """梗塞巣マスクから体積・重症度を算出する。"""
    img  = nib.load(mask_path)
    data = img.get_fdata()
    zooms = img.header.get_zooms()
    vox_vol = zooms[0] * zooms[1] * zooms[2]
    voxels  = int(np.sum(data > 0.5))
    vol_ml  = voxels * vox_vol / 1000.0

    severity = dysphagia_risk = "不明"
    for lo, hi, sev, risk in DEEPISLES_SEVERITY:
        if lo <= vol_ml < hi:
            severity = sev
            dysphagia_risk = risk
            break

    # 重心（MNI空間でおおよその位置）
    if voxels > 0:
        coords = np.array(np.where(data > 0.5))
        centroid_vox = coords.mean(axis=1)
        affine = img.affine
        centroid_mm = (affine[:3, :3] @ centroid_vox + affine[:3, 3]).tolist()
    else:
        centroid_mm = [0.0, 0.0, 0.0]

    return {
        "lesion_volume_ml":       round(vol_ml, 3),
        "lesion_voxels":          voxels,
        "severity":               severity,
        "dysphagia_risk_from_lesion": dysphagia_risk,
        "centroid_mm":            [round(c, 1) for c in centroid_mm],
        "mask_path":              mask_path,
        "method":                 "DeepISLES (DWI+ADC+FLAIR MRI ensemble)",
    }


def run_pipeline(
    dwi_path: str,
    adc_path: str,
    flair_path: str,
    output_dir: str,
    patient_id: str,
    skull_strip: bool = True,
    fast: bool = False,
    dwi_dicom: str = None,
    adc_dicom: str = None,
    flair_dicom: str = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # DICOMがある場合はNIfTI変換
    with tempfile.TemporaryDirectory() as tmpdir:
        if dwi_dicom:
            dwi_path = dicom_to_nifti(dwi_dicom, os.path.join(tmpdir, "dwi.nii.gz"))
        if adc_dicom:
            adc_path = dicom_to_nifti(adc_dicom, os.path.join(tmpdir, "adc.nii.gz"))
        if flair_dicom:
            flair_path = dicom_to_nifti(flair_dicom, os.path.join(tmpdir, "flair.nii.gz"))

        mask_path = run_deepisles(dwi_path, adc_path, flair_path,
                                  output_dir, skull_strip, fast)
        result = quantify_infarct(mask_path)

    result["patient_id"] = patient_id

    # 予測API用にマスクを標準位置にコピー
    standard_mask = os.path.join(output_dir, "infarct.nii.gz")
    if mask_path != standard_mask:
        import shutil
        shutil.copy(mask_path, standard_mask)
        result["mask_path"] = standard_mask

    out_json = os.path.join(output_dir, f"{patient_id}_deepisles_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[DeepISLES] 結果保存: {out_json}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="DeepISLES 梗塞巣セグメンテーション")
    parser.add_argument("--dwi",   help="DWI NIfTIパス")
    parser.add_argument("--adc",   help="ADC NIfTIパス")
    parser.add_argument("--flair", help="FLAIR NIfTIパス")
    parser.add_argument("--dwi-dicom",   help="DWI DICOMディレクトリ")
    parser.add_argument("--adc-dicom",   help="ADC DICOMディレクトリ")
    parser.add_argument("--flair-dicom", help="FLAIR DICOMディレクトリ")
    parser.add_argument("--output",      required=True, help="出力ディレクトリ")
    parser.add_argument("--patient-id",  default="unknown")
    parser.add_argument("--skull-strip", action="store_true", default=True)
    parser.add_argument("--fast",        action="store_true", help="高速モード（単一モデル）")
    args = parser.parse_args()

    if not any([args.dwi, args.dwi_dicom]):
        parser.error("--dwi または --dwi-dicom が必要です")

    run_pipeline(
        dwi_path=args.dwi or "",
        adc_path=args.adc or "",
        flair_path=args.flair or "",
        output_dir=args.output,
        patient_id=args.patient_id,
        skull_strip=args.skull_strip,
        fast=args.fast,
        dwi_dicom=args.dwi_dicom,
        adc_dicom=args.adc_dicom,
        flair_dicom=args.flair_dicom,
    )


if __name__ == "__main__":
    main()
