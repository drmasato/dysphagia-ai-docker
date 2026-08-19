#!/bin/bash
# ISLES2024 NCCT 梗塞巣 nnU-Net v2 学習スクリプト
#
# 使用方法:
#   bash train_stroke_ct.sh [--folds <0-4|all>] [--config <3d_fullres|3d_lowres>]
#
# 前提:
#   - /hdd/nnUNet_raw/Dataset500_StrokeCT/ が準備済み
#   - CUDA が利用可能
#   - GPU VRAM >= 8GB (3d_fullres) または >= 4GB (3d_lowres)

set -euo pipefail

HDD="/media/morita/ubuntuHDD/nnunet_stroke"
DATASET_ID=500
CONFIG="3d_fullres"
FOLDS="all"
TRAINER="nnUNetTrainer"

while [[ $# -gt 0 ]]; do
    case $1 in
        --folds)  FOLDS="$2";  shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "不明なオプション: $1"; exit 1 ;;
    esac
done

echo "======================================="
echo " nnU-Net v2 脳卒中CT学習"
echo " Dataset: ${DATASET_ID}"
echo " Config:  ${CONFIG}"
echo " Folds:   ${FOLDS}"
echo "======================================="

docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e nnUNet_raw="${HDD}/nnUNet_raw" \
  -e nnUNet_preprocessed="${HDD}/nnUNet_preprocessed" \
  -e nnUNet_results="${HDD}/nnUNet_results" \
  -v "${HDD}:${HDD}" \
  dysphagia-nnunet:latest \
  bash -c "
    echo '--- Step 1: データ整合性チェック ---'
    nnUNetv2_plan_and_preprocess \
      -d ${DATASET_ID} \
      -c ${CONFIG} \
      --verify_dataset_integrity \
      -np 4 -npp 4

    echo ''
    echo '--- Step 2: 学習開始 ---'
    if [ '${FOLDS}' = 'all' ]; then
        for fold in 0 1 2 3 4; do
            echo \"=== Fold \${fold}/4 ===\"
            nnUNetv2_train ${DATASET_ID} ${CONFIG} \${fold} \
              --npz \
              -num_gpus 1
        done
    else
        nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLDS} \
          --npz \
          -num_gpus 1
    fi

    echo ''
    echo '--- Step 3: ベストモデルの選択 ---'
    nnUNetv2_find_best_configuration ${DATASET_ID} -c ${CONFIG}

    echo ''
    echo '======================================='
    echo ' 学習完了！'
    echo ' モデル保存先: ${HDD}/nnUNet_results/Dataset${DATASET_ID}_StrokeCT/'
    echo '======================================='
  " 2>&1 | tee "${HDD}/training_$(date +%Y%m%d_%H%M%S).log"
