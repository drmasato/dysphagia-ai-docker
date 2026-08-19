"""
CPAISD (Zenodo 10892316) → nnU-Net v2 Dataset500 変換スクリプト。

データ構造:
  dataset/{train|val}/{UID}/{slice_num}/
    image.npz   : CT スライス (512×512, float64, HU値)
    mask.npz    : ラベル (512×512, uint8, 0=背景/1=core/2=penumbra)
    raw.dcm     : DICOM（PixelSpacing取得用）
    metadata.json: {"slice_location": float}

出力 (nnU-Net Dataset500):
  Dataset500_StrokeCT/
    imagesTr/stroke_XXXX_0000.nii.gz
    labelsTr/stroke_XXXX.nii.gz
    imagesTs/stroke_val_XXXX_0000.nii.gz
    dataset.json

使用方法:
  python3 prepare_isles2024.py \
    --zip /hdd/nnunet_stroke/isles2024_raw/dataset.zip \
    --output /hdd/nnunet_stroke/nnUNet_raw/Dataset500_StrokeCT \
    [--merge-labels]   # core+penumbra→1クラス（推奨）
"""

import argparse
import io
import json
import os
import re
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom


# ──────────────────────────────────────────────────────────────────────────────

def get_spacing_from_dicom(dcm_bytes: bytes) -> tuple[float, float]:
    """DICOMバイト列からPixelSpacingを取得する。"""
    try:
        ds = pydicom.dcmread(io.BytesIO(dcm_bytes), stop_before_pixels=True)
        ps = ds.get("PixelSpacing", [0.9, 0.9])
        return float(ps[0]), float(ps[1])
    except Exception:
        return 0.9, 0.9  # デフォルト: 0.9mm


def build_volume(
    slices: list[tuple[float, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    (slice_location, image_2d, mask_2d) リストから3Dボリュームを構築する。
    戻り値: (ct_volume, mask_volume, slice_thickness)
    """
    # slice_location でソート
    slices.sort(key=lambda x: x[0])
    locs   = [s[0] for s in slices]
    images = [s[1] for s in slices]
    masks  = [s[2] for s in slices]

    # スライス厚を推定
    if len(locs) > 1:
        diffs = [abs(locs[i+1] - locs[i]) for i in range(len(locs)-1)]
        slice_thickness = float(np.median(diffs))
    else:
        slice_thickness = 5.0

    ct_vol   = np.stack(images, axis=-1).astype(np.float32)   # (512,512,N)
    mask_vol = np.stack(masks,  axis=-1).astype(np.uint8)     # (512,512,N)
    return ct_vol, mask_vol, slice_thickness


def save_nifti(array: np.ndarray, spacing: tuple[float, float, float], path: str):
    """numpy配列をNIfTIとして保存する。"""
    dx, dy, dz = spacing
    affine = np.diag([dx, dy, dz, 1.0])
    img = nib.Nifti1Image(array, affine)
    img.header.set_zooms(spacing)
    nib.save(img, path)


def convert_cpaisd(
    zip_path: str,
    output_dir: str,
    merge_labels: bool = True,
    val_ratio: float = 0.2,
    split: str = "train",
):
    """
    CPAISD ZIPをnnU-Net Dataset500形式に変換する。

    Parameters
    ----------
    zip_path    : dataset.zip のパス
    output_dir  : Dataset500_StrokeCT の出力先
    merge_labels: True → core(1)+penumbra(2) を infarct(1) にマージ
    split       : "train" または "val"
    """
    out = Path(output_dir)
    images_tr = out / "imagesTr"
    labels_tr  = out / "labelsTr"
    images_ts  = out / "imagesTs"
    for d in [images_tr, labels_tr, images_ts]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"ZIPを開いています: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_entries = zf.namelist()

        # UID一覧を収集
        uid_set = {}
        for e in all_entries:
            m = re.match(r"dataset/(train|val)/([^/]+)/", e)
            if m:
                sp, uid = m.group(1), m.group(2)
                uid_set.setdefault(sp, set()).add(uid)

        train_uids = sorted(uid_set.get("train", []))
        val_uids   = sorted(uid_set.get("val",   []))
        print(f"学習症例: {len(train_uids)}件, 検証症例: {len(val_uids)}件")

        n_additional_val = max(0, int(len(train_uids) * val_ratio) - len(val_uids))
        if n_additional_val > 0:
            extra_val    = train_uids[-n_additional_val:]
            train_uids   = train_uids[:-n_additional_val]
            val_uids     = val_uids + extra_val
            print(f"  → 学習から{n_additional_val}件を検証に移動: 学習{len(train_uids)}, 検証{len(val_uids)}")

        def process_case(uid: str, sp: str) -> tuple[np.ndarray, np.ndarray, tuple]:
            """1症例のスライス群を3Dボリュームに変換する。"""
            prefix = f"dataset/{sp}/{uid}/"
            case_entries = [e for e in all_entries if e.startswith(prefix) and "/" in e[len(prefix):]]
            slice_nums = set(e[len(prefix):].split("/")[0] for e in case_entries if e[len(prefix):])

            slices = []
            spacing_xy = (0.9, 0.9)

            for sn in sorted(slice_nums):
                img_key  = f"{prefix}{sn}/image.npz"
                mask_key = f"{prefix}{sn}/mask.npz"
                meta_key = f"{prefix}{sn}/metadata.json"
                dcm_key  = f"{prefix}{sn}/raw.dcm"

                if img_key not in all_entries or mask_key not in all_entries:
                    continue

                # スライス位置
                slice_loc = float(sn)  # デフォルト
                if meta_key in all_entries:
                    with zf.open(meta_key) as f:
                        meta = json.load(f)
                    slice_loc = float(meta.get("slice_location", sn))

                # CT画像
                with zf.open(img_key) as f:
                    img_arr = np.load(io.BytesIO(f.read()))["image"].astype(np.float32)

                # マスク
                with zf.open(mask_key) as f:
                    mask_arr = np.load(io.BytesIO(f.read()))["mask"].astype(np.uint8)

                # PixelSpacing（最初のスライスのDICOMから取得）
                if len(slices) == 0 and dcm_key in all_entries:
                    with zf.open(dcm_key) as f:
                        spacing_xy = get_spacing_from_dicom(f.read())

                slices.append((slice_loc, img_arr, mask_arr))

            if not slices:
                return None, None, None

            ct_vol, mask_vol, dz = build_volume(slices)

            if merge_labels:
                mask_vol = (mask_vol > 0).astype(np.uint8)

            spacing = (spacing_xy[0], spacing_xy[1], dz)
            return ct_vol, mask_vol, spacing

        # 学習セット変換
        print(f"\n学習セット変換中...")
        converted_train = 0
        for i, uid in enumerate(train_uids, 1):
            case_id = f"stroke_{i:04d}"
            ct, mask, spacing = process_case(uid, "train")
            if ct is None:
                print(f"  スキップ（データなし）: {uid[:20]}")
                continue

            save_nifti(ct,   spacing, str(images_tr / f"{case_id}_0000.nii.gz"))
            save_nifti(mask, spacing, str(labels_tr  / f"{case_id}.nii.gz"))
            converted_train += 1

            infarct_voxels = int((mask > 0).sum())
            print(f"  [{i:3d}/{len(train_uids)}] {case_id}: "
                  f"shape={ct.shape} spacing={tuple(round(s,2) for s in spacing)} "
                  f"infarct={infarct_voxels}vox")

        # 検証セット変換
        print(f"\n検証セット変換中...")
        converted_val = 0
        for i, uid in enumerate(val_uids, 1):
            sp = "val" if uid in sorted(uid_set.get("val", [])) else "train"
            case_id = f"stroke_val_{i:04d}"
            ct, mask, spacing = process_case(uid, sp)
            if ct is None:
                continue
            save_nifti(ct, spacing, str(images_ts / f"{case_id}_0000.nii.gz"))
            converted_val += 1
            print(f"  [{i:2d}/{len(val_uids)}] {case_id}: shape={ct.shape}")

    # dataset.json
    labels = {"background": 0, "infarct": 1} if merge_labels else \
             {"background": 0, "infarct_core": 1, "penumbra": 2}

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": converted_train,
        "file_ending": ".nii.gz",
        "name": "Dataset500_StrokeCT",
        "description": (
            "Ischemic stroke CT lesion segmentation. "
            "Source: CPAISD (Zenodo 10892316, CC BY 4.0). "
            f"Labels: {'merged (core+penumbra=infarct)' if merge_labels else 'separate (core/penumbra)'}"
        ),
        "reference": "https://zenodo.org/records/10892316",
        "licence": "CC BY 4.0",
        "release": "1.0",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    json_path = out / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"""
========================================
 変換完了
  学習: {converted_train} 症例
  検証: {converted_val} 症例
  ラベル: {'infarct（マージ）' if merge_labels else 'core + penumbra（2クラス）'}
  出力: {output_dir}

 次のステップ:
  bash train_stroke_ct.sh
========================================""")


def main():
    parser = argparse.ArgumentParser(
        description="CPAISD (ISLES2024) → nnU-Net Dataset500 変換"
    )
    parser.add_argument("--zip", required=True,
                        help="dataset.zip のパス")
    parser.add_argument("--output", required=True,
                        help="出力先（例: /hdd/nnUNet_raw/Dataset500_StrokeCT）")
    parser.add_argument("--merge-labels", action="store_true", default=True,
                        help="core+penumbra を1クラス infarct にマージ（推奨、デフォルトON）")
    parser.add_argument("--no-merge-labels", dest="merge_labels", action="store_false",
                        help="core と penumbra を別クラスとして学習する")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="学習データからの検証分割比率（デフォルト: 0.2）")
    args = parser.parse_args()

    convert_cpaisd(
        zip_path=args.zip,
        output_dir=args.output,
        merge_labels=args.merge_labels,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
