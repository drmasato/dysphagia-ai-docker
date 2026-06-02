#!/bin/bash
# FSL BIANCA 白質病変自動セグメンテーションパイプライン
# 入力: FLAIR画像（必須）、T1画像（オプション）
# 出力: WMHバイナリマスク、確率マップ

set -euo pipefail

usage() {
    echo "使用方法: $0 --flair <path> --output <dir> [--t1 <path>] [--patient-id <id>]"
    echo ""
    echo "オプション:"
    echo "  --flair       FLAIR NIFTIファイルパス（必須）"
    echo "  --output      出力ディレクトリ（必須）"
    echo "  --t1          T1 NIFTIファイルパス（オプション）"
    echo "  --patient-id  患者ID（デフォルト: unknown）"
    echo "  --bianca-threshold  BIANCA確率閾値（デフォルト: 0.9）"
    exit 1
}

# デフォルト値
FLAIR=""
OUTPUT_DIR=""
T1=""
PATIENT_ID="unknown"
BIANCA_THRESHOLD=0.9

while [[ $# -gt 0 ]]; do
    case $1 in
        --flair)          FLAIR="$2";             shift 2 ;;
        --output)         OUTPUT_DIR="$2";        shift 2 ;;
        --t1)             T1="$2";                shift 2 ;;
        --patient-id)     PATIENT_ID="$2";        shift 2 ;;
        --bianca-threshold) BIANCA_THRESHOLD="$2"; shift 2 ;;
        --help|-h)        usage ;;
        *)                echo "不明なオプション: $1"; usage ;;
    esac
done

[[ -z "$FLAIR" ]] && echo "エラー: --flair が必要です" && usage
[[ -z "$OUTPUT_DIR" ]] && echo "エラー: --output が必要です" && usage

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=== FSL BIANCA パイプライン開始: ${PATIENT_ID} ==="
echo "FLAIR: $FLAIR"
echo "T1: ${T1:-なし}"
echo "出力: $OUTPUT_DIR"

# --------------------------------------------------------
# Step 1: FLAIR脳抽出（BET）
# --------------------------------------------------------
echo ""
echo "--- Step 1: BET 脳抽出 ---"
FLAIR_BRAIN="${OUTPUT_DIR}/flair_brain"
bet "$FLAIR" "$FLAIR_BRAIN" -f 0.3 -R -m
FLAIR_BRAIN_NII="${FLAIR_BRAIN}.nii.gz"
FLAIR_MASK="${FLAIR_BRAIN}_mask.nii.gz"
echo "BET完了: $FLAIR_BRAIN_NII"

# --------------------------------------------------------
# Step 2: T1がある場合はFLAIR空間に位置合わせ
# --------------------------------------------------------
T1_IN_FLAIR_SPACE=""
if [[ -n "$T1" ]]; then
    echo ""
    echo "--- Step 2: T1→FLAIR空間 位置合わせ (FLIRT) ---"
    T1_BRAIN="${OUTPUT_DIR}/t1_brain"
    bet "$T1" "$T1_BRAIN" -f 0.35 -R
    T1_TO_FLAIR="${OUTPUT_DIR}/t1_to_flair"
    flirt \
        -in "${T1_BRAIN}.nii.gz" \
        -ref "$FLAIR_BRAIN_NII" \
        -out "$T1_TO_FLAIR" \
        -omat "${T1_TO_FLAIR}.mat" \
        -dof 6 \
        -cost corratio
    T1_IN_FLAIR_SPACE="${T1_TO_FLAIR}.nii.gz"
    echo "位置合わせ完了: $T1_IN_FLAIR_SPACE"
fi

# --------------------------------------------------------
# Step 3: FLAIR→MNI標準空間への変換行列計算
# --------------------------------------------------------
echo ""
echo "--- Step 3: FLAIR→MNI 変換行列計算 ---"
MNI_TEMPLATE="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"
FLAIR_TO_MNI="${OUTPUT_DIR}/flair_to_mni"
flirt \
    -in "$FLAIR_BRAIN_NII" \
    -ref "$MNI_TEMPLATE" \
    -out "$FLAIR_TO_MNI" \
    -omat "${FLAIR_TO_MNI}.mat" \
    -dof 12 \
    -cost corratio
echo "MNI変換行列: ${FLAIR_TO_MNI}.mat"

# --------------------------------------------------------
# Step 4: BIANCAマスターファイル作成
# --------------------------------------------------------
echo ""
echo "--- Step 4: BIANCAマスターファイル作成 ---"
MASTER_FILE="${OUTPUT_DIR}/bianca_master.txt"

if [[ -n "$T1_IN_FLAIR_SPACE" ]]; then
    echo "${FLAIR_BRAIN_NII} ${T1_IN_FLAIR_SPACE} ${FLAIR_TO_MNI}.mat" > "$MASTER_FILE"
else
    echo "${FLAIR_BRAIN_NII} ${FLAIR_BRAIN_NII} ${FLAIR_TO_MNI}.mat" > "$MASTER_FILE"
fi
echo "マスターファイル: $MASTER_FILE"

# --------------------------------------------------------
# Step 5: BIANCA実行
# --------------------------------------------------------
echo ""
echo "--- Step 5: BIANCA 実行 ---"
BIANCA_OUTPUT="${OUTPUT_DIR}/bianca_output"

bianca \
    --singlefile="$MASTER_FILE" \
    --labelfeaturenum=1 \
    --brainmaskfeaturenum=1 \
    --querysubjectnum=1 \
    --spatialweight=2 \
    --patchsizes=3 \
    -o "$BIANCA_OUTPUT" \
    --savefeatureimages \
    --saveprobmap

BIANCA_PROBMAP="${BIANCA_OUTPUT}.nii.gz"
echo "BIANCA確率マップ: $BIANCA_PROBMAP"

# --------------------------------------------------------
# Step 6: バイナリマスク変換・体積計算
# --------------------------------------------------------
echo ""
echo "--- Step 6: 二値化・体積計算 ---"
WMH_MASK="${OUTPUT_DIR}/wmh_mask.nii.gz"
fslmaths "$BIANCA_PROBMAP" -thr "$BIANCA_THRESHOLD" -bin "$WMH_MASK"

# 体積計算（voxel数 × voxelサイズ）
VOXEL_COUNT=$(fslstats "$WMH_MASK" -V | awk '{print $1}')
VOXEL_SIZE=$(fslval "$FLAIR_BRAIN_NII" pixdim1)
VOXEL_SIZE_Y=$(fslval "$FLAIR_BRAIN_NII" pixdim2)
VOXEL_SIZE_Z=$(fslval "$FLAIR_BRAIN_NII" pixdim3)
VOXEL_VOL_MM3=$(echo "$VOXEL_SIZE * $VOXEL_SIZE_Y * $VOXEL_SIZE_Z" | bc -l)
WMH_VOL_ML=$(echo "scale=4; $VOXEL_COUNT * $VOXEL_VOL_MM3 / 1000" | bc -l)

echo "WMH体積: ${WMH_VOL_ML} mL"

# 結果をJSON出力
RESULT_JSON="${OUTPUT_DIR}/${PATIENT_ID}_bianca_raw.json"
cat > "$RESULT_JSON" <<EOF
{
  "patient_id": "${PATIENT_ID}",
  "wmh_voxel_count": ${VOXEL_COUNT},
  "wmh_volume_ml": ${WMH_VOL_ML},
  "bianca_threshold": ${BIANCA_THRESHOLD},
  "wmh_mask_path": "${WMH_MASK}",
  "bianca_probmap_path": "${BIANCA_PROBMAP}"
}
EOF

echo ""
echo "=== BIANCA パイプライン完了 ==="
echo "結果JSON: $RESULT_JSON"

# quantify_wmh.pyでFazekas分類・詳細定量化
python3 /workspace/quantify_wmh.py \
    --wmh-mask "$WMH_MASK" \
    --output-dir "$OUTPUT_DIR" \
    --patient-id "$PATIENT_ID" \
    --wmh-volume-ml "$WMH_VOL_ML"
