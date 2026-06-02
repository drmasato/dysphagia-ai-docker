.PHONY: all build-api build-full start-full start-api stop \
        generate-data generate-nifti train quickstart quickstart-gpu \
        test-api test-pipeline pipeline-status logs logs-all shell-api clean help

COMPOSE    = docker compose -f pipeline/docker-compose.yml
DATA_DIR   = data
MODEL_DIR  = xgboost-shap/models
PATIENT_ID ?= PT-TEST-001

help:
	@echo "=== 嚥下障害AI Dockerパイプライン ==="
	@echo ""
	@echo "セットアップ:"
	@echo "  make quickstart      - CPU専用（APIのみ）クイックスタート"
	@echo "  make quickstart-gpu  - GPU全パイプライン起動（推奨）"
	@echo ""
	@echo "ビルド:"
	@echo "  make build-api       - 予測APIのみビルド"
	@echo "  make build-full      - 全コンテナビルド（nnU-Net/TotalSeg/BIANCA/API）"
	@echo ""
	@echo "起動・停止:"
	@echo "  make start-api       - 予測APIサーバーのみ起動（ポート8000）"
	@echo "  make start-full      - 全サービス起動（GPU必要）"
	@echo "  make stop            - 全サービス停止"
	@echo ""
	@echo "学習・データ:"
	@echo "  make generate-data   - 合成学習データ生成（200症例CSV）"
	@echo "  make generate-nifti  - テスト用NIfTI画像生成（CT/FLAIR/T1）"
	@echo "  make train           - XGBoostモデル学習（Optuna最適化）"
	@echo ""
	@echo "テスト・確認:"
	@echo "  make test-api        - ヘルスチェック + テスト予測"
	@echo "  make test-pipeline   - TotalSegmentatorで嚥下筋定量化テスト"
	@echo "  make pipeline-status - 全コンテナ稼働状況確認"
	@echo "  make logs            - 予測APIログ表示"
	@echo "  make logs-all        - 全コンテナログ表示"
	@echo ""
	@echo "Swagger UI: http://localhost:8000/docs"

# ── クイックスタート ─────────────────────────────────────────

quickstart: build-api generate-data train start-api
	@echo ""
	@echo "=== CPU クイックスタート完了 ==="
	@echo "Swagger UI: http://localhost:8000/docs"
	@sleep 3 && $(MAKE) test-api

quickstart-gpu: build-full generate-data generate-nifti train start-full
	@echo ""
	@echo "=== GPU 全パイプライン起動完了 ==="
	@echo "Swagger UI:       http://localhost:8000/docs"
	@echo "パイプライン状態: http://localhost:8000/pipeline/status"
	@sleep 5 && $(MAKE) test-api

# ── ビルド ────────────────────────────────────────────────────

build-api:
	@echo "--- 予測APIイメージをビルド中 ---"
	docker build -t dysphagia-predictor:latest ./xgboost-shap/

build-full:
	@echo "--- 全コンテナイメージをビルド中（並列） ---"
	docker build -t dysphagia-nnunet:latest    ./nnunet/         &
	docker build -t dysphagia-totalseg:latest  ./totalsegmentator/ &
	docker build -t dysphagia-bianca:latest    ./fsl-bianca/     &
	docker build -t dysphagia-predictor:latest ./xgboost-shap/   &
	docker build -t dysphagia-pipeline:latest  -f ./pipeline/Dockerfile.pipeline ./pipeline/ &
	wait
	@echo "--- 全イメージビルド完了 ---"
	@docker images | grep dysphagia

# ── 起動・停止 ────────────────────────────────────────────────

start-api:
	@echo "--- 予測APIサーバー起動 ---"
	@docker rm -f dysphagia-predictor 2>/dev/null || true
	docker run -d \
	  --name dysphagia-predictor \
	  --restart unless-stopped \
	  -p 8000:8000 \
	  -v $(PWD)/$(MODEL_DIR):/workspace/models \
	  -v $(PWD)/$(DATA_DIR):/workspace/data \
	  -v /var/run/docker.sock:/var/run/docker.sock \
	  -e MODEL_PATH=/workspace/models/dysphagia_xgb_model.joblib \
	  -e RESULTS_DIR=/workspace/data/output \
	  -e DATA_INPUT_DIR=/workspace/data/input \
	  -e HOST_DATA_DIR=$(PWD)/$(DATA_DIR) \
	  dysphagia-predictor:latest
	@sleep 4 && curl -s http://localhost:8000/health | python3 -m json.tool

start-full:
	@echo "--- 全サービス起動（GPU） ---"
	@docker rm -f dysphagia-predictor 2>/dev/null || true
	$(COMPOSE) --profile full up -d
	@sleep 6 && curl -s http://localhost:8000/health | python3 -m json.tool

stop:
	@echo "--- 全サービス停止 ---"
	$(COMPOSE) --profile full down
	@docker rm -f dysphagia-predictor 2>/dev/null || true

# ── データ・学習 ──────────────────────────────────────────────

generate-data:
	@echo "--- 合成学習データ生成（200症例）---"
	@mkdir -p $(DATA_DIR)
	docker run --rm \
	  -v $(PWD)/$(DATA_DIR):/workspace/data \
	  -v $(PWD)/scripts:/scripts \
	  dysphagia-predictor:latest \
	  python3 /scripts/generate_sample_data.py \
	    --n-samples 200 \
	    --output /workspace/data/training_data.csv

generate-nifti:
	@echo "--- テスト用NIfTI画像生成（CT/FLAIR/T1）---"
	@mkdir -p $(DATA_DIR)/input
	docker run --rm \
	  -v $(PWD)/$(DATA_DIR):/data \
	  -v $(PWD)/scripts:/scripts \
	  dysphagia-predictor:latest \
	  bash -c "pip install nibabel -q && python3 /scripts/generate_test_nifti.py \
	    --output-dir /data/input \
	    --patient-id $(PATIENT_ID)"

train:
	@echo "--- XGBoostモデル学習（Optuna 50試行 + 5-fold CV）---"
	@mkdir -p $(MODEL_DIR)
	docker run --rm \
	  -v $(PWD)/$(DATA_DIR):/workspace/data \
	  -v $(PWD)/$(MODEL_DIR):/workspace/models \
	  dysphagia-predictor:latest \
	  python3 /workspace/train_model.py \
	    --data /workspace/data/training_data.csv \
	    --output-dir /workspace/models \
	    --n-trials 50

# ── テスト・確認 ──────────────────────────────────────────────

test-api:
	@echo "=== API ヘルスチェック ==="
	@curl -s http://localhost:8000/health | python3 -m json.tool
	@echo ""
	@echo "=== テスト予測（72歳男性・NIHSS 8・心房細動あり）==="
	@curl -s -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d '{"patient_id":"PT-TEST-001","aspects_score":7,"lesion_volume_ml":12.5,"wmh_volume_ml":3.2,"tongue_csa_cm2":3.8,"superior_pharyngeal_constrictor_volume_cm3":1.8,"hotspot_affected_count":2,"nihss_score":8,"age":72,"albumin_gdl":3.5,"bmi":21.0,"prior_stroke":0,"atrial_fibrillation":1,"diabetes":0,"sex_male":1}' \
	  | python3 -m json.tool

test-totalseg:
	@echo "=== TotalSegmentator 嚥下筋定量化テスト ==="
	@mkdir -p $(DATA_DIR)/output/totalseg
	docker run --rm --gpus all \
	  -v $(PWD)/$(DATA_DIR)/input:/input:ro \
	  -v $(PWD)/$(DATA_DIR)/output/totalseg:/output \
	  dysphagia-totalseg:latest \
	  python3 /workspace/quantify_muscles.py \
	    --input /input/$(PATIENT_ID)_CT.nii.gz \
	    --output /output \
	    --patient-id $(PATIENT_ID) \
	    --sex male

test-pipeline:
	@echo "=== パイプライン統合テスト (API経由) ==="
	@curl -s -X POST http://localhost:8000/pipeline/run \
	  -H "Content-Type: application/json" \
	  -d '{"patient_id":"$(PATIENT_ID)","ct_nifti_filename":"$(PATIENT_ID)_CT.nii.gz","flair_nifti_filename":"$(PATIENT_ID)_FLAIR.nii.gz","t1_nifti_filename":"$(PATIENT_ID)_T1.nii.gz","sex":"male","age":72,"aspects_score":7,"nihss_score":8,"albumin_gdl":3.5,"bmi":21.0,"prior_stroke":0,"atrial_fibrillation":1,"diabetes":0}' \
	  | python3 -m json.tool

pipeline-status:
	@echo "=== コンテナ稼働状況 ==="
	@curl -s http://localhost:8000/pipeline/status | python3 -m json.tool

logs:
	docker logs -f dysphagia-predictor

logs-all:
	$(COMPOSE) --profile full logs -f

shell-api:
	docker exec -it dysphagia-predictor bash

# ── クリーン ──────────────────────────────────────────────────

clean:
	@echo "--- 全コンテナ・イメージ・データ削除 ---"
	$(COMPOSE) --profile full down -v --remove-orphans 2>/dev/null || true
	docker rm -f dysphagia-predictor 2>/dev/null || true
	docker rmi dysphagia-nnunet dysphagia-totalseg dysphagia-bianca \
	           dysphagia-predictor dysphagia-pipeline 2>/dev/null || true
	rm -rf $(DATA_DIR)/output $(MODEL_DIR)
	@echo "クリーンアップ完了"
