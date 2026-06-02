# 頭部CT × 嚥下機能 画像AI自動定量パイプライン

脳卒中急性期患者を対象に、頭部CT所見から嚥下障害リスクを自動定量化するAIパイプライン。

> **規制上の注意**: 本システムは研究用途専用です。臨床診断への直接使用には薬機法上のSaMD承認が必要です。

## アーキテクチャ

```
[DICOM入力]
    ↓ pydicom / SimpleITK
[nnU-Net v2]         → 梗塞巣体積・部位マスク
[TotalSegmentator]   → 嚥下筋断面積・体積
[FSL BIANCA]         → WMH体積・Fazekas分類
    ↓
[XGBoost + SHAP]     → 嚥下回復確率・リスク分類・SHAP説明
    ↓
[FastAPI REST API]   → JSONレポート
```

## ファイル構成

```
dysphagia-ai-docker/
├── Makefile
├── README.md
├── nnunet/
│   ├── Dockerfile
│   └── inference.py
├── totalsegmentator/
│   ├── Dockerfile
│   └── quantify_muscles.py
├── fsl-bianca/
│   ├── Dockerfile
│   ├── bianca_pipeline.sh
│   └── quantify_wmh.py
├── xgboost-shap/
│   ├── Dockerfile
│   ├── train_model.py
│   ├── predict.py
│   └── api_server.py
├── pipeline/
│   ├── docker-compose.yml
│   ├── Dockerfile.pipeline
│   └── run_pipeline.py
└── scripts/
    └── generate_sample_data.py
```

## クイックスタート（GPU不要・約20分）

```bash
cd dysphagia-ai-docker

# ビルド → データ生成 → 学習 → 起動
make quickstart
```

## 予測APIの使用例

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PT-001",
    "aspects_score": 7,
    "lesion_volume_ml": 12.5,
    "wmh_volume_ml": 3.2,
    "tongue_csa_cm2": 3.8,
    "hotspot_affected_count": 2,
    "nihss_score": 8,
    "age": 72,
    "albumin_gdl": 3.5,
    "atrial_fibrillation": 1,
    "sex_male": 1
  }'
```

## Swagger UI

`http://localhost:8000/docs` でインタラクティブAPIドキュメントを参照可能。

## ハードウェア要件

| 構成 | GPU | RAM | 用途 |
|---|---|---|---|
| 最小（APIのみ） | 不要 | 16 GB | 予測API・モデル学習 |
| 推奨（全機能） | RTX 3060 12GB+ | 32 GB | 全パイプライン |

## リスク分類基準

| 回復確率 | リスク | 推奨アクション |
|---|---|---|
| < 30% | 高リスク | VE/VF当日実施・ST緊急紹介 |
| 30–70% | 中リスク | RSST・MWST後にST紹介 |
| > 70% | 低リスク | 経過観察（3日後再評価） |

## 参考文献

- Isensee et al. Nat Methods 2021 (nnU-Net)
- Wasserthal et al. Radiol Artif Intell 2023 (TotalSegmentator)
- Griffanti et al. NeuroImage 2016 (BIANCA)
- Tian et al. BMC Med Inform Decis Mak 2025 (XGBoost)
