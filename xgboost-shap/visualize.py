"""
CT + 梗塞巣マスク・筋肉マスクの多断面可視化モジュール。
nibabel + matplotlib で HTML/PNG を生成し FastAPI から配信する。
"""

import base64
import io
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np


# ── カラーマップ定義 ──────────────────────────────────────────────────────────

MASK_COLORS = {
    "infarct":                          ([1.0, 0.2, 0.2], "梗塞巣"),
    "intracerebral_hemorrhage":         ([1.0, 0.5, 0.0], "頭蓋内出血"),
    "tongue":                           ([0.2, 0.8, 0.2], "舌筋"),
    "superior_pharyngeal_constrictor":  ([0.0, 0.6, 1.0], "上咽頭収縮筋"),
    "middle_pharyngeal_constrictor":    ([0.0, 0.4, 0.8], "中咽頭収縮筋"),
    "inferior_pharyngeal_constrictor":  ([0.0, 0.2, 0.6], "下咽頭収縮筋"),
    "masseter_left":                    ([1.0, 0.8, 0.0], "咬筋（左）"),
    "masseter_right":                   ([1.0, 0.6, 0.0], "咬筋（右）"),
    "sternocleidomastoid_left":         ([0.8, 0.0, 0.8], "胸鎖乳突筋（左）"),
    "sternocleidomastoid_right":        ([0.6, 0.0, 0.6], "胸鎖乳突筋（右）"),
    "brain":                            ([0.9, 0.9, 0.7], "脳"),
}


def _load_nifti(path: str) -> Optional[np.ndarray]:
    """NIFTIを読み込んでnumpy配列を返す。失敗時はNone。"""
    if not path or not os.path.exists(path):
        return None
    try:
        img = nib.load(path)
        return img.get_fdata()
    except Exception:
        return None


def _normalize_ct(ct: np.ndarray, wc: float = 40, ww: float = 80) -> np.ndarray:
    """CT画像をウィンドウ設定でグレースケール正規化する。"""
    lo = wc - ww / 2
    hi = wc + ww / 2
    ct_norm = np.clip(ct, lo, hi)
    return (ct_norm - lo) / (hi - lo)


def _get_center_slice(data: np.ndarray, axis: int) -> int:
    """最大シグナルを含むスライスインデックスを返す。"""
    if data.max() == 0:
        return data.shape[axis] // 2
    sums = np.sum(data > 0, axis=tuple(i for i in range(3) if i != axis))
    return int(np.argmax(sums))


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def generate_multiview(
    ct_path: str,
    mask_paths: dict[str, str],
    patient_id: str,
    wc: float = 40,
    ww: float = 80,
) -> str:
    """
    CT + 複数マスクを 軸位・矢状・冠状の3断面で可視化し base64 PNG を返す。

    Parameters
    ----------
    ct_path     : CT NIfTI ファイルパス
    mask_paths  : {"infarct": path, "masseter_left": path, ...}
    patient_id  : 患者ID（タイトル表示用）
    wc, ww      : CT Window Center / Width（脳設定: 40/80）
    """
    ct = _load_nifti(ct_path)
    if ct is None:
        return ""

    ct_norm = _normalize_ct(ct, wc, ww)

    # マスクデータ読み込み
    masks = {}
    for name, path in mask_paths.items():
        data = _load_nifti(path)
        if data is not None and data.max() > 0:
            masks[name] = data

    # 各断面の中心スライスを決定（梗塞巣があればその中心、なければ脳の中心）
    ref_mask = masks.get("infarct", list(masks.values())[0] if masks else None)
    ax_idx  = _get_center_slice(ref_mask, 2) if ref_mask is not None else ct.shape[2] // 2
    sag_idx = _get_center_slice(ref_mask, 0) if ref_mask is not None else ct.shape[0] // 2
    cor_idx = _get_center_slice(ref_mask, 1) if ref_mask is not None else ct.shape[1] // 2

    views = [
        ("軸位断 Axial",   ct_norm[:, :, ax_idx].T,  {k: v[:, :, ax_idx].T  for k, v in masks.items()}),
        ("矢状断 Sagittal", ct_norm[sag_idx, :, :].T, {k: v[sag_idx, :, :].T for k, v in masks.items()}),
        ("冠状断 Coronal",  ct_norm[:, cor_idx, :].T, {k: v[:, cor_idx, :].T for k, v in masks.items()}),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"患者: {patient_id}  |  脳窓 WC={wc} WW={ww}", color="white",
                 fontsize=13, y=1.02)

    legend_patches = []
    for ax, (title, ct_slice, mask_slices) in zip(axes, views):
        ax.set_facecolor("black")
        ax.imshow(ct_slice, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")

        for mask_name, mask_slice in mask_slices.items():
            color, label = MASK_COLORS.get(mask_name, ([1, 0, 0], mask_name))
            rgba = np.zeros((*mask_slice.shape, 4))
            rgba[..., :3] = color
            rgba[..., 3] = np.where(mask_slice > 0.5, 0.5, 0)
            ax.imshow(rgba, origin="lower", interpolation="nearest")
            if not any(p.get_label() == label for p in legend_patches):
                legend_patches.append(
                    mpatches.Patch(color=color, label=label, alpha=0.8)
                )

        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")

    if legend_patches:
        fig.legend(handles=legend_patches, loc="lower center", ncol=min(len(legend_patches), 4),
                   framealpha=0.3, labelcolor="white", facecolor="#1a1a2e",
                   fontsize=9, bbox_to_anchor=(0.5, -0.08))

    return _fig_to_base64(fig)


def generate_axial_montage(
    ct_path: str,
    mask_path: Optional[str],
    patient_id: str,
    n_slices: int = 12,
    wc: float = 40,
    ww: float = 80,
    mask_name: str = "infarct",
) -> str:
    """
    軸位断のモンタージュ（N枚並べ）を生成する。
    梗塞巣マスクがある場合はそのスライス範囲を中心に表示。
    """
    ct = _load_nifti(ct_path)
    if ct is None:
        return ""
    ct_norm = _normalize_ct(ct, wc, ww)
    mask = _load_nifti(mask_path) if mask_path else None

    total_slices = ct.shape[2]
    if mask is not None and mask.max() > 0:
        mask_z = np.where(np.any(mask > 0, axis=(0, 1)))[0]
        z_min, z_max = max(0, mask_z.min() - 5), min(total_slices - 1, mask_z.max() + 5)
    else:
        z_min, z_max = total_slices // 4, 3 * total_slices // 4

    slice_indices = np.linspace(z_min, z_max, n_slices, dtype=int)

    cols = 4
    rows = (n_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.patch.set_facecolor("#1a1a2e")
    axes = axes.flatten()

    color, label = MASK_COLORS.get(mask_name, ([1, 0, 0], mask_name))

    for i, (ax, z) in enumerate(zip(axes, slice_indices)):
        ax.set_facecolor("black")
        ax.imshow(ct_norm[:, :, z].T, cmap="gray", vmin=0, vmax=1,
                  origin="lower", interpolation="nearest")
        if mask is not None and mask[:, :, z].max() > 0:
            rgba = np.zeros((*ct_norm[:, :, z].T.shape, 4))
            rgba[..., :3] = color
            rgba[..., 3] = np.where(mask[:, :, z].T > 0.5, 0.55, 0)
            ax.imshow(rgba, origin="lower", interpolation="nearest")
        ax.set_title(f"z={z}", color="white", fontsize=8)
        ax.axis("off")

    for ax in axes[len(slice_indices):]:
        ax.axis("off")

    fig.suptitle(f"{patient_id} – 軸位断モンタージュ ({label})", color="white", fontsize=12)
    return _fig_to_base64(fig)


def generate_html_report(
    ct_path: str,
    mask_paths: dict[str, str],
    patient_id: str,
    clinical_summary: dict = None,
    wc: float = 40,
    ww: float = 80,
) -> str:
    """全断面 + モンタージュ + 臨床サマリーを含む HTML レポートを生成する。"""

    multiview_b64 = generate_multiview(ct_path, mask_paths, patient_id, wc, ww)

    # 梗塞巣があればモンタージュも生成
    infarct_path = mask_paths.get("infarct")
    montage_b64  = generate_axial_montage(ct_path, infarct_path, patient_id,
                                          wc=wc, ww=ww) if infarct_path else ""

    # 臨床サマリーHTML
    summary_rows = ""
    if clinical_summary:
        for k, v in clinical_summary.items():
            summary_rows += f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"

    multiview_html = (
        f'<img src="data:image/png;base64,{multiview_b64}" style="width:100%">'
        if multiview_b64 else "<p>CT画像が見つかりません</p>"
    )
    montage_html = (
        f'<h2>軸位断モンタージュ（梗塞巣範囲）</h2>'
        f'<img src="data:image/png;base64,{montage_b64}" style="width:100%">'
        if montage_b64 else ""
    )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CT可視化レポート – {patient_id}</title>
<style>
  body {{ background:#1a1a2e; color:#e0e0e0; font-family:sans-serif; margin:0; padding:20px; }}
  h1 {{ color:#4fc3f7; border-bottom:2px solid #4fc3f7; padding-bottom:8px; }}
  h2 {{ color:#81c784; margin-top:30px; }}
  .card {{ background:#16213e; border-radius:8px; padding:16px; margin:16px 0; }}
  table {{ border-collapse:collapse; width:100%; max-width:600px; }}
  td {{ padding:8px 12px; border-bottom:1px solid #2a2a4a; }}
  td:first-child {{ color:#90caf9; width:40%; }}
  .disclaimer {{ color:#ff8a65; font-size:0.85em; border:1px solid #ff8a65;
                 padding:10px; border-radius:4px; margin-top:20px; }}
  img {{ max-width:100%; border-radius:6px; display:block; margin:10px auto; }}
</style>
</head>
<body>
<h1>CT可視化レポート</h1>
<div class="card">
  <table>
    <tr><td>患者ID</td><td><strong>{patient_id}</strong></td></tr>
    {summary_rows}
  </table>
</div>

<div class="card">
  <h2>3断面表示（軸位・矢状・冠状）</h2>
  {multiview_html}
</div>

{f'<div class="card">{montage_html}</div>' if montage_html else ''}

<div class="disclaimer">
  ⚠ 本レポートは研究目的のAI出力です。臨床判断は医師が行ってください。
  SaMD承認なし・臨床診断への直接使用不可。
</div>
</body>
</html>"""
