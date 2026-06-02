"""
テスト用合成NIfTI脳CT・FLAIR・T1画像を生成するスクリプト。
実際のDICOMデータがない環境でパイプラインの動作確認に使用する。
"""

import argparse
import os

import nibabel as nib
import numpy as np


def ellipsoid_mask(shape, center, radii):
    """ベクトル演算で楕円体マスクを高速生成する。"""
    cx, cy, cz = center
    rx, ry, rz = radii
    xx, yy, zz = np.mgrid[:shape[0], :shape[1], :shape[2]]
    dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2
    return dist <= 1.0


def generate_ct_brain(shape=(192, 192, 128), voxel_size=(1.0, 1.0, 1.25)):
    """合成脳CT画像（HU値）を生成する。"""
    rng = np.random.default_rng(42)
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    data = np.zeros(shape, dtype=np.float32)

    brain_mask  = ellipsoid_mask(shape, (cx, cy, cz), (60, 51, 45))
    skull_outer = ellipsoid_mask(shape, (cx, cy, cz), (70, 60, 52))
    skull_mask  = skull_outer & ~ellipsoid_mask(shape, (cx, cy, cz), (63, 54, 47))

    # 脳実質: 20–45 HU
    n_brain = brain_mask.sum()
    data[brain_mask] = rng.normal(35, 8, n_brain).clip(10, 60)

    # 頭蓋骨: 600–800 HU
    n_skull = skull_mask.sum()
    data[skull_mask] = rng.uniform(600, 800, n_skull)

    # 疑似梗塞巣（低吸収域 5–20 HU）: 左 MCA 領域
    infarct = ellipsoid_mask(shape, (cx + 20, cy - 10, cz), (15, 12, 10)) & brain_mask
    n_inf = infarct.sum()
    data[infarct] = rng.uniform(5, 20, n_inf)

    affine = np.diag([*voxel_size, 1.0])
    return nib.Nifti1Image(data, affine)


def generate_flair_brain(shape=(192, 192, 128), voxel_size=(1.0, 1.0, 1.25)):
    """合成FLAIR画像（WMH高信号あり）を生成する。"""
    rng = np.random.default_rng(123)
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    data = np.zeros(shape, dtype=np.float32)

    brain_mask = ellipsoid_mask(shape, (cx, cy, cz), (60, 51, 45))
    n_brain = brain_mask.sum()
    data[brain_mask] = rng.normal(500, 80, n_brain).clip(200, 900)

    # 脳室周囲 WMH（4箇所）
    for dx, dy in [(-15, -15), (15, -15), (-15, 15), (15, 15)]:
        wmh = ellipsoid_mask(shape, (cx + dx, cy + dy, cz), (8, 6, 5)) & brain_mask
        n_wmh = wmh.sum()
        if n_wmh > 0:
            data[wmh] = rng.uniform(1200, 1800, n_wmh)

    affine = np.diag([*voxel_size, 1.0])
    return nib.Nifti1Image(data, affine)


def generate_t1_brain(shape=(192, 192, 128), voxel_size=(1.0, 1.0, 1.25)):
    """合成T1画像を生成する。"""
    rng = np.random.default_rng(456)
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    data = np.zeros(shape, dtype=np.float32)

    brain_mask = ellipsoid_mask(shape, (cx, cy, cz), (60, 51, 45))
    n_brain = brain_mask.sum()
    data[brain_mask] = rng.normal(800, 100, n_brain).clip(400, 1200)

    affine = np.diag([*voxel_size, 1.0])
    return nib.Nifti1Image(data, affine)


def main():
    parser = argparse.ArgumentParser(description="テスト用NIfTIデータ生成")
    parser.add_argument("--output-dir", default="/data/input", help="出力ディレクトリ")
    parser.add_argument("--patient-id", default="PT-TEST-001", help="患者ID")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for name, gen_fn, suffix in [
        ("脳CT",   generate_ct_brain,    "_CT.nii.gz"),
        ("FLAIR",  generate_flair_brain, "_FLAIR.nii.gz"),
        ("T1",     generate_t1_brain,    "_T1.nii.gz"),
    ]:
        print(f"合成{name}生成中 (192×192×128)...")
        img = gen_fn()
        path = os.path.join(args.output_dir, f"{args.patient_id}{suffix}")
        nib.save(img, path)
        print(f"  保存: {path}")

    print(f"\n生成完了: {args.output_dir}")


if __name__ == "__main__":
    main()
