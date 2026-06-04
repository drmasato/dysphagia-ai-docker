"""
ISLES 2024 NCCTデータセットをnnU-Net v2フォーマットに変換するスクリプト。

データセット: CPAISD (Zenodo 10892316) - 149症例 CC BY 4.0
  NCCT + CTA + CTP + 梗塞巣マスク（DWI由来ラベル）

使用方法:
  1. データダウンロード
     wget "https://zenodo.org/records/10892316/files/dataset.zip"
     unzip dataset.zip -d /data/isles2024_raw

  2. nnU-Net形式に変換
     python3 prepare_isles2024.py \
       --input /data/isles2024_raw \
       --output /data/nnUNet_raw/Dataset500_StrokeCT \
       --modality ncct
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import SimpleITK as sitk


def convert_to_nnunet_format(
    raw_dir: str,
    output_dir: str,
    modality: str = "ncct",
    val_ratio: float = 0.2,
):
    """
    ISLES 2024 raw データを nnU-Net Dataset フォーマットに変換する。

    nnU-Net Dataset500 構成:
      Dataset500_StrokeCT/
        imagesTr/  ← 学習CT（_0000.nii.gz）
        labelsTr/  ← 学習マスク（.nii.gz）
        imagesTs/  ← 検証CT（_0000.nii.gz）
        dataset.json
    """
    raw_path = Path(raw_dir)
    out_path  = Path(output_dir)
    images_tr = out_path / "imagesTr"
    labels_tr = out_path / "labelsTr"
    images_ts = out_path / "imagesTs"

    for d in [images_tr, labels_tr, images_ts]:
        d.mkdir(parents=True, exist_ok=True)

    # 入力ディレクトリからケースを列挙
    modality_suffix_map = {
        "ncct":  "**/ncct*.nii*",
        "flair": "**/flair*.nii*",
        "dwi":   "**/dwi*.nii*",
    }
    pattern = modality_suffix_map.get(modality, "**/*.nii*")
    ct_files = sorted(raw_path.rglob(pattern.lstrip("**/").split("*")[0]))

    # ケースIDベースで対応するラベルファイルを探す
    cases = []
    for ct_file in raw_path.rglob("*.nii.gz"):
        if modality.lower() in ct_file.name.lower():
            # ラベルファイルを同ディレクトリ内で検索
            label_candidates = list(ct_file.parent.glob("*lesion*")) + \
                               list(ct_file.parent.glob("*mask*")) + \
                               list(ct_file.parent.glob("*label*"))
            if label_candidates:
                cases.append((ct_file, label_candidates[0]))

    if not cases:
        print(f"警告: {raw_dir} に {modality} ファイルが見つかりません。")
        print("データ構造を確認して BIDS形式かどうか確認してください。")
        _try_bids_format(raw_path, out_path, images_tr, labels_tr, images_ts, modality, val_ratio)
        return

    n_val = max(1, int(len(cases) * val_ratio))
    train_cases = cases[:-n_val]
    val_cases   = cases[-n_val:]

    print(f"変換開始: 学習 {len(train_cases)}件, 検証 {len(val_cases)}件")

    for i, (ct, label) in enumerate(train_cases, 1):
        case_id = f"stroke_{i:04d}"
        shutil.copy(ct, images_tr / f"{case_id}_0000.nii.gz")
        shutil.copy(label, labels_tr / f"{case_id}.nii.gz")
        print(f"  [学習] {case_id}: {ct.name}")

    for i, (ct, label) in enumerate(val_cases, 1):
        case_id = f"stroke_val_{i:04d}"
        shutil.copy(ct, images_ts / f"{case_id}_0000.nii.gz")
        print(f"  [検証] {case_id}: {ct.name}")

    _write_dataset_json(out_path, len(train_cases), len(val_cases))
    print(f"\nnnU-Net形式変換完了: {output_dir}")
    print(f"次のステップ:")
    print(f"  nnUNetv2_plan_and_preprocess -d 500 -c 3d_fullres")
    print(f"  nnUNetv2_train 500 3d_fullres all --npz")


def _try_bids_format(raw_path, out_path, images_tr, labels_tr, images_ts, modality, val_ratio):
    """BIDS形式（ISLES 2024標準）での変換を試みる。"""
    print("BIDS形式として解析を試みます...")

    # ISLES 2024 BIDS: sub-XXX/ses-YYY/anat/
    ncct_files = list(raw_path.rglob("*ncct*.nii.gz")) + \
                 list(raw_path.rglob("*NCCT*.nii.gz")) + \
                 list(raw_path.rglob("*ct.nii.gz"))

    # デリバティブのラベル（梗塞巣マスク）
    label_files = list(raw_path.rglob("*lesion*.nii.gz")) + \
                  list(raw_path.rglob("*infarct*.nii.gz")) + \
                  list((raw_path / "derivatives").rglob("*.nii.gz") if (raw_path / "derivatives").exists() else [])

    print(f"  CT候補: {len(ncct_files)}件")
    print(f"  ラベル候補: {len(label_files)}件")

    if not ncct_files:
        print("  データが見つかりません。手動でディレクトリ構造を確認してください:")
        for p in list(raw_path.iterdir())[:5]:
            print(f"    {p}")
        return

    # ラベルとCTをサブジェクトIDで対応付け
    cases = []
    for ct in ncct_files:
        sub_id = next((p for p in ct.parts if p.startswith("sub-")), None)
        if sub_id:
            matching_labels = [l for l in label_files if sub_id in str(l)]
            if matching_labels:
                cases.append((ct, matching_labels[0]))

    if cases:
        n_val = max(1, int(len(cases) * val_ratio))
        for i, (ct, lbl) in enumerate(cases[:-n_val], 1):
            cid = f"stroke_{i:04d}"
            shutil.copy(ct, images_tr / f"{cid}_0000.nii.gz")
            shutil.copy(lbl, labels_tr / f"{cid}.nii.gz")
        for i, (ct, _) in enumerate(cases[-n_val:], 1):
            shutil.copy(ct, images_ts / f"stroke_val_{i:04d}_0000.nii.gz")
        _write_dataset_json(out_path, len(cases) - n_val, n_val)
        print(f"BIDS変換完了: {len(cases)}症例")


def _write_dataset_json(out_path: Path, n_train: int, n_val: int):
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "infarct": 1},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": "Dataset500_StrokeCT",
        "description": "Ischemic stroke lesion segmentation (NCCT) - ISLES 2024 / local data",
        "reference": "https://zenodo.org/records/10892316",
        "licence": "CC BY 4.0",
        "release": "1.0",
    }
    with open(out_path / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="ISLES 2024 → nnU-Net形式変換")
    parser.add_argument("--input",    required=True, help="ISLES 2024 raw ディレクトリ")
    parser.add_argument("--output",   required=True, help="nnU-Net Dataset500 出力先")
    parser.add_argument("--modality", default="ncct", choices=["ncct", "flair", "dwi"])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()
    convert_to_nnunet_format(args.input, args.output, args.modality, args.val_ratio)


if __name__ == "__main__":
    main()
