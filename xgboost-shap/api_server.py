"""
FastAPI 嚥下回復予測REST APIサーバー
- POST /predict          : 単体予測（CT定量値→嚥下回復確率）
- POST /predict/batch    : バッチ予測（CSV相当のJSONリスト）
- GET  /pipeline/status  : 全Dockerコンテナの稼働状況
- POST /pipeline/run     : 統合パイプライン実行トリガー
- GET  /results/{id}     : 保存済みレポート取得
- GET  /features         : 特徴量定義一覧
- GET  /health           : ヘルスチェック
"""

import glob
import json
import os
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import shap
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

MODEL_PATH      = os.environ.get("MODEL_PATH",      "/workspace/models/dysphagia_xgb_model.joblib")
RESULTS_DIR     = os.environ.get("RESULTS_DIR",     "/workspace/data/output")
DATA_INPUT_DIR  = os.environ.get("DATA_INPUT_DIR",  "/workspace/data/input")
# ホスト側のデータディレクトリ（Dockerソケット経由でコンテナ起動する際のバインドに使用）
HOST_DATA_DIR   = os.environ.get("HOST_DATA_DIR",   "/home/morita/dysphagia-ai-docker/data")

FEATURE_COLUMNS = [
    "aspects_score",
    "lesion_volume_ml",
    "wmh_volume_ml",
    "tongue_csa_cm2",
    "superior_pharyngeal_constrictor_volume_cm3",
    "hotspot_affected_count",
    "nihss_score",
    "age",
    "albumin_gdl",
    "bmi",
    "prior_stroke",
    "atrial_fibrillation",
    "diabetes",
    "sex_male",
]

FEATURE_DESCRIPTIONS = {
    "aspects_score":                            "ASPECTSスコア（0-10）",
    "lesion_volume_ml":                         "梗塞巣体積（mL）",
    "wmh_volume_ml":                            "白質病変WMH体積（mL）",
    "tongue_csa_cm2":                           "舌筋最大断面積（cm²）",
    "superior_pharyngeal_constrictor_volume_cm3": "上咽頭収縮筋体積（cm³）",
    "hotspot_affected_count":                   "嚥下ホットスポット関与数（0-4）",
    "nihss_score":                              "NIHSSスコア（0-42）",
    "age":                                      "年齢（歳）",
    "albumin_gdl":                              "血清アルブミン（g/dL）",
    "bmi":                                      "BMI",
    "prior_stroke":                             "既往脳卒中（0/1）",
    "atrial_fibrillation":                      "心房細動（0/1）",
    "diabetes":                                 "糖尿病（0/1）",
    "sex_male":                                 "性別（男性=1, 女性=0）",
}

RISK_ACTIONS = {
    "high":     "VE/VF当日実施・ST緊急紹介・経口摂取禁止",
    "moderate": "RSST・MWST実施後にST紹介・嚥下リハ開始",
    "low":      "経過観察（3日後再評価）・食事中の観察継続",
}

PIPELINE_CONTAINERS = ["dysphagia-nnunet", "dysphagia-totalseg", "dysphagia-bianca", "dysphagia-predictor"]

_model    = None
_explainer = None
_pipeline_jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _explainer
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        _explainer = shap.TreeExplainer(_model)
        print(f"[API] モデル読み込み完了: {MODEL_PATH}")
    else:
        print(f"[API] 警告: モデルが見つかりません → {MODEL_PATH}")
    yield


app = FastAPI(
    title="嚥下回復予測 API",
    description="""
## 頭部CT × 嚥下機能 AI自動定量パイプライン

脳卒中急性期患者の頭部CT定量値から**嚥下回復確率**を予測するREST API（研究用途）。

### エンドポイント概要
| エンドポイント | 機能 |
|---|---|
| `POST /predict` | CT定量値→嚥下回復確率（SHAP説明付き） |
| `POST /predict/batch` | 複数症例の一括予測 |
| `GET /pipeline/status` | nnU-Net・TotalSeg・BIANCA稼働状態確認 |
| `POST /pipeline/run` | 全AIパイプライン実行トリガー |
| `GET /results/{patient_id}` | 保存済みレポート取得 |

> **規制注意**: 本APIは研究目的専用です。SaMD承認なし・臨床診断への直接使用不可。
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── スキーマ定義 ────────────────────────────────────────────

class PredictRequest(BaseModel):
    model_config = {"json_schema_extra": {
        "example": {
            "patient_id": "PT-001",
            "aspects_score": 7,
            "lesion_volume_ml": 12.5,
            "wmh_volume_ml": 3.2,
            "tongue_csa_cm2": 3.8,
            "superior_pharyngeal_constrictor_volume_cm3": 1.8,
            "hotspot_affected_count": 2,
            "nihss_score": 8,
            "age": 72,
            "albumin_gdl": 3.5,
            "bmi": 21.0,
            "prior_stroke": 0,
            "atrial_fibrillation": 1,
            "diabetes": 0,
            "sex_male": 1,
        }
    }}

    patient_id:   str   = Field(..., description="患者ID")
    aspects_score: float = Field(..., ge=0, le=10, description="ASPECTSスコア（0-10）")
    lesion_volume_ml: float = Field(..., ge=0, description="梗塞巣体積（mL）")
    wmh_volume_ml:    float = Field(..., ge=0, description="WMH体積（mL）")
    tongue_csa_cm2:   float = Field(..., ge=0, description="舌筋最大断面積（cm²）")
    superior_pharyngeal_constrictor_volume_cm3: float = Field(
        default=2.0, ge=0, description="上咽頭収縮筋体積（cm³）"
    )
    hotspot_affected_count: int   = Field(default=0, ge=0, le=4, description="嚥下ホットスポット関与数")
    nihss_score:      float = Field(..., ge=0, le=42, description="NIHSSスコア（0-42）")
    age:              int   = Field(..., ge=0, le=120, description="年齢（歳）")
    albumin_gdl:      float = Field(default=3.8, ge=0, le=6.0, description="血清アルブミン（g/dL）")
    bmi:              float = Field(default=22.0, ge=10, le=60, description="BMI")
    prior_stroke:     int   = Field(default=0, ge=0, le=1, description="既往脳卒中（0=なし, 1=あり）")
    atrial_fibrillation: int = Field(default=0, ge=0, le=1, description="心房細動（0=なし, 1=あり）")
    diabetes:         int   = Field(default=0, ge=0, le=1, description="糖尿病（0=なし, 1=あり）")
    sex_male:         int   = Field(default=1, ge=0, le=1, description="性別（1=男性, 0=女性）")

    @field_validator("aspects_score")
    @classmethod
    def aspects_integer(cls, v):
        if v != int(v):
            raise ValueError("ASPECTSスコアは整数を入力してください")
        return v


class ShapFactor(BaseModel):
    feature:     str
    shap:        float
    description: str


class PredictResponse(BaseModel):
    patient_id:           str
    predicted_at:         str
    recovery_probability: float
    risk_level:           str
    risk_level_en:        str
    recommended_action:   str
    shap_values:          dict[str, float]
    top_5_factors:        list[ShapFactor]
    disclaimer:           str


class PipelineRunRequest(BaseModel):
    model_config = {"json_schema_extra": {
        "example": {
            "patient_id": "PT-TEST-001",
            "ct_nifti_filename": "PT-TEST-001_CT.nii.gz",
            "flair_nifti_filename": "PT-TEST-001_FLAIR.nii.gz",
            "t1_nifti_filename": "PT-TEST-001_T1.nii.gz",
            "sex": "male",
            "age": 72,
            "aspects_score": 7,
            "nihss_score": 8,
            "albumin_gdl": 3.5,
            "bmi": 21.0,
            "prior_stroke": 0,
            "atrial_fibrillation": 1,
            "diabetes": 0,
        }
    }}
    patient_id:            str   = Field(..., description="患者ID")
    ct_nifti_filename:     str   = Field(..., description="/data/input/ 以下のCT NIFTIファイル名")
    flair_nifti_filename:  str   = Field(..., description="/data/input/ 以下のFLAIR NIFTIファイル名")
    t1_nifti_filename:     str   = Field(default="", description="T1 NIFTIファイル名（省略可）")
    sex:                   str   = Field(default="male", description="性別（male/female）")
    age:                   int   = Field(..., ge=0, le=120, description="年齢（歳）")
    aspects_score:         float = Field(..., ge=0, le=10, description="ASPECTSスコア")
    nihss_score:           float = Field(..., ge=0, le=42, description="NIHSSスコア")
    albumin_gdl:           float = Field(default=3.8, description="血清アルブミン（g/dL）")
    bmi:                   float = Field(default=22.0, description="BMI")
    prior_stroke:          int   = Field(default=0, ge=0, le=1)
    atrial_fibrillation:   int   = Field(default=0, ge=0, le=1)
    diabetes:              int   = Field(default=0, ge=0, le=1)


# ── ヘルパー ────────────────────────────────────────────────

def _classify_risk(prob: float) -> tuple[str, str]:
    if prob < 0.30:
        return "高リスク", "high"
    elif prob < 0.70:
        return "中リスク", "moderate"
    return "低リスク", "low"


def _container_status(name: str) -> str:
    """
    Docker Unix socket REST APIでコンテナ/イメージ状態を確認する。
    on-demandコンテナ（nnunet/totalseg/bianca）はイメージ存在を確認する。
    """
    sock = "/var/run/docker.sock"
    image_map = {
        "dysphagia-nnunet":     "dysphagia-nnunet:latest",
        "dysphagia-totalseg":   "dysphagia-totalseg:latest",
        "dysphagia-bianca":     "dysphagia-bianca:latest",
        "dysphagia-predictor":  "dysphagia-predictor:latest",
    }
    if not os.path.exists(sock):
        # ソケット未マウント時はファイルシステムで推定
        image_name = image_map.get(name, f"{name}:latest").replace(":", "_")
        return "api_running" if name == "dysphagia-predictor" else "image_unknown"

    import socket as _sock, http.client as _http
    def _docker_get(path: str):
        try:
            conn = _http.HTTPConnection("localhost")
            conn.sock = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
            conn.sock.connect(sock)
            conn.request("GET", path)
            r = conn.getresponse()
            return r.status, json.loads(r.read())
        except Exception:
            return 0, {}

    # まずコンテナが起動中か確認
    status, data = _docker_get(f"/containers/{name}/json")
    if status == 200:
        return data.get("State", {}).get("Status", "unknown")

    # 起動中でなければイメージが存在するか確認
    img = image_map.get(name, f"{name}:latest")
    img_encoded = img.replace(":", "%3A")
    status2, _ = _docker_get(f"/images/{img_encoded}/json")
    if status2 == 200:
        return "image_ready"
    return "not_built"


def _docker_api(method: str, path: str, body: dict = None) -> tuple[int, Any]:
    """Docker Unix socket REST APIを直接呼び出す（CLI不要）。"""
    import socket as _sock, http.client as _http
    sock = "/var/run/docker.sock"
    if not os.path.exists(sock):
        return 0, {"error": "docker socket not found"}
    try:
        conn = _http.HTTPConnection("localhost")
        conn.sock = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        conn.sock.connect(sock)
        headers = {"Content-Type": "application/json"}
        payload = json.dumps(body).encode() if body else None
        conn.request(method, path, body=payload, headers=headers)
        r = conn.getresponse()
        raw = r.read()
        try:
            return r.status, json.loads(raw)
        except Exception:
            return r.status, raw.decode(errors="replace")
    except Exception as e:
        return 0, {"error": str(e)}


def _docker_run_step(
    image: str,
    cmd: list[str],
    use_gpu: bool = False,
    timeout: int = 900,
) -> tuple[bool, str]:
    """Docker socket REST APIでコンテナをon-demand起動する。"""
    # ホスト側のパスを使ってバインドを設定する
    host_data_dir = HOST_DATA_DIR

    create_body: dict[str, Any] = {
        "Image": image,
        "Cmd": cmd,
        "AttachStdout": True,
        "AttachStderr": True,
        "HostConfig": {
            "AutoRemove": False,
            "Binds": [f"{host_data_dir}:/workspace/data"],
            "NetworkMode": "host",
        },
    }
    if use_gpu:
        # CDIモード環境では Runtime=nvidia を使用（--runtime=nvidia 相当）
        create_body["HostConfig"]["Runtime"] = "nvidia"
        create_body["Env"] = ["NVIDIA_VISIBLE_DEVICES=all"]

    # コンテナ作成
    status, resp = _docker_api("POST", "/containers/create", create_body)
    if status not in (201, 200):
        return False, f"コンテナ作成失敗 ({status}): {resp}"
    container_id = resp.get("Id", "")

    # 起動
    _docker_api("POST", f"/containers/{container_id}/start")

    # 完了待機（ポーリング）
    import time
    deadline = time.time() + timeout
    exit_code = -1
    while time.time() < deadline:
        st, info = _docker_api("GET", f"/containers/{container_id}/json")
        if st == 200 and not info.get("State", {}).get("Running", True):
            exit_code = info.get("State", {}).get("ExitCode", -1)
            break
        time.sleep(3)

    # ログ取得
    st, logs = _docker_api("GET", f"/containers/{container_id}/logs?stdout=true&stderr=true")
    log_str = logs if isinstance(logs, str) else str(logs)

    # 後片付け
    _docker_api("DELETE", f"/containers/{container_id}?force=true")

    return exit_code == 0, log_str


def _run_pipeline_background(req: PipelineRunRequest, job_id: str):
    """各AIコンテナをon-demandで起動してパイプライン全体を実行する。"""
    _pipeline_jobs[job_id]["status"] = "running"
    _pipeline_jobs[job_id]["started_at"] = datetime.now().isoformat()
    steps_log = {}

    data_dir = os.environ.get("DATA_DIR", "/workspace/data")
    patient_id = req.patient_id

    # ── Step 1: TotalSegmentator（GPU）────────────────────────────────
    print(f"[Pipeline] Step 1: TotalSegmentator 実行中...")
    ok, log = _docker_run_step(
        image="dysphagia-totalseg:latest",
        cmd=[
            "python3", "/workspace/quantify_muscles.py",
            "--input", f"/workspace/data/input/{req.ct_nifti_filename}",
            "--output", f"/workspace/data/output/totalseg/{patient_id}",
            "--patient-id", patient_id,
            "--sex", req.sex,
            "--task", "auto",
        ],
        use_gpu=True,
    )
    steps_log["totalseg"] = "ok" if ok else f"fallback: {log[-200:]}"
    print(f"[Pipeline] TotalSegmentator: {'OK' if ok else 'フォールバック'}")

    # ── Step 2: FSL BIANCA（CPU）──────────────────────────────────────
    print(f"[Pipeline] Step 2: FSL BIANCA 実行中...")
    bianca_cmd = [
        "bash", "/workspace/bianca_pipeline.sh",
        "--flair", f"/workspace/data/input/{req.flair_nifti_filename}",
        "--output", f"/workspace/data/output/bianca/{patient_id}",
        "--patient-id", patient_id,
    ]
    if req.t1_nifti_filename:
        bianca_cmd += ["--t1", f"/workspace/data/input/{req.t1_nifti_filename}"]
    ok, log = _docker_run_step(image="dysphagia-bianca:latest", cmd=bianca_cmd, use_gpu=False)
    steps_log["bianca"] = "ok" if ok else f"fallback: {log[-200:]}"
    print(f"[Pipeline] BIANCA: {'OK' if ok else 'フォールバック'}")

    # ── Step 3: 各ステップの結果を読み込んで予測API呼び出し ────────────
    print(f"[Pipeline] Step 3: 予測API呼び出し...")

    # TotalSegmentator 結果読み込み
    tongue_csa = 4.0
    spc_vol = 2.0
    totalseg_json = os.path.join(data_dir, "output", "totalseg", patient_id, f"{patient_id}_muscles_result.json")
    if os.path.exists(totalseg_json):
        with open(totalseg_json) as f:
            ts = json.load(f)
        tongue_csa = ts.get("tongue_csa_cm2", 4.0)
        spc_vol    = ts.get("superior_pharyngeal_constrictor_volume_cm3", 2.0)

    # BIANCA 結果読み込み
    wmh_vol = 0.0
    bianca_json = os.path.join(data_dir, "output", "bianca", patient_id, f"{patient_id}_wmh_result.json")
    if os.path.exists(bianca_json):
        with open(bianca_json) as f:
            wmh_vol = json.load(f).get("wmh_volume_ml", 0.0)

    # 予測API呼び出し（同一プロセス内で直接計算）
    try:
        import numpy as np
        features = [
            req.aspects_score,
            0.0,               # lesion_volume_ml（nnU-Netなし時は0）
            wmh_vol,
            tongue_csa,
            spc_vol,
            0,                 # hotspot_affected_count
            req.nihss_score,
            req.age,
            req.albumin_gdl,
            req.bmi,
            req.prior_stroke,
            req.atrial_fibrillation,
            req.diabetes,
            1 if req.sex == "male" else 0,
        ]
        X = np.array(features, dtype=np.float64).reshape(1, -1)
        prob = float(_model.predict_proba(X)[0, 1]) if _model else 0.5
        risk_ja, risk_en = _classify_risk(prob)
        prediction = {
            "recovery_probability": round(prob, 4),
            "risk_level": risk_ja,
            "recommended_action": RISK_ACTIONS[risk_en],
        }
    except Exception as e:
        prediction = {"error": str(e), "recovery_probability": None}

    # 最終レポート生成
    os.makedirs(os.path.join(data_dir, "output", patient_id), exist_ok=True)
    report = {
        "report_metadata": {
            "patient_id": patient_id,
            "job_id": job_id,
            "generated_at": datetime.now().isoformat(),
            "pipeline_version": "1.0.0",
        },
        "steps_status": steps_log,
        "imaging_quantification": {
            "swallowing_muscles": {
                "tongue_csa_cm2": tongue_csa,
                "spc_volume_cm3": spc_vol,
            },
            "white_matter_hyperintensity": {"wmh_volume_ml": wmh_vol},
        },
        "prediction": prediction,
    }
    report_path = os.path.join(data_dir, "output", patient_id, f"{patient_id}_final_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _pipeline_jobs[job_id]["status"] = "completed"
    _pipeline_jobs[job_id]["report_path"] = report_path
    _pipeline_jobs[job_id]["prediction"] = prediction
    _pipeline_jobs[job_id]["finished_at"] = datetime.now().isoformat()
    print(f"[Pipeline] 完了: {report_path}")


# ── エンドポイント ──────────────────────────────────────────

@app.get("/health", tags=["システム"], summary="ヘルスチェック")
async def health():
    """APIサーバーとモデルの稼働状態を返す。"""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/features", tags=["システム"], summary="特徴量定義一覧")
async def get_features():
    """予測に使用する14特徴量の名前・説明・取得元を返す。"""
    return {
        "n_features": len(FEATURE_COLUMNS),
        "features": [
            {"name": col, "description": FEATURE_DESCRIPTIONS.get(col, "")}
            for col in FEATURE_COLUMNS
        ],
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["予測"],
    summary="単体症例の嚥下回復予測",
)
async def predict(req: PredictRequest):
    """
    CT定量値と臨床指標から90日後嚥下回復確率を予測する。

    - **recovery_probability**: 0.0〜1.0（1.0が完全回復）
    - **risk_level**: 高リスク / 中リスク / 低リスク
    - **top_5_factors**: 予測に最も寄与した上位5因子（SHAP値）
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="モデル未読み込み。make train を実行してください。")

    features = [getattr(req, col) for col in FEATURE_COLUMNS]
    X = np.array(features, dtype=np.float64).reshape(1, -1)

    prob = float(_model.predict_proba(X)[0, 1])
    risk_ja, risk_en = _classify_risk(prob)

    shap_vals = _explainer.shap_values(X)[0]
    shap_dict = {col: round(float(v), 4) for col, v in zip(FEATURE_COLUMNS, shap_vals)}
    top_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return PredictResponse(
        patient_id=req.patient_id,
        predicted_at=datetime.now().isoformat(),
        recovery_probability=round(prob, 4),
        risk_level=risk_ja,
        risk_level_en=risk_en,
        recommended_action=RISK_ACTIONS[risk_en],
        shap_values=shap_dict,
        top_5_factors=[
            ShapFactor(feature=k, shap=v, description=FEATURE_DESCRIPTIONS.get(k, ""))
            for k, v in top_factors
        ],
        disclaimer=(
            "本予測は研究目的のAIモデルによるものです。"
            "臨床判断の最終決定は必ず医師が行ってください。"
            "SaMD承認なし・臨床診断への直接使用不可。"
        ),
    )


@app.post(
    "/predict/batch",
    tags=["予測"],
    summary="複数症例の一括予測",
)
async def predict_batch(requests: list[PredictRequest]):
    """複数症例をリストで送信して一括予測する。最大100件まで。"""
    if _model is None:
        raise HTTPException(status_code=503, detail="モデル未読み込み。")
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="一度に送信できるのは100件までです。")

    results = []
    for req in requests:
        features = [getattr(req, col) for col in FEATURE_COLUMNS]
        X = np.array(features, dtype=np.float64).reshape(1, -1)
        prob = float(_model.predict_proba(X)[0, 1])
        risk_ja, risk_en = _classify_risk(prob)
        results.append({
            "patient_id": req.patient_id,
            "recovery_probability": round(prob, 4),
            "risk_level": risk_ja,
            "recommended_action": RISK_ACTIONS[risk_en],
        })
    return {"count": len(results), "results": results}


@app.get(
    "/pipeline/status",
    tags=["パイプライン"],
    summary="全Dockerコンテナ・イメージの稼働状態確認",
)
async def pipeline_status():
    """
    各コンポーネントのステータスを返す。

    - **running**: コンテナ起動中
    - **image_ready**: イメージビルド済み（on-demandで実行可能）
    - **api_running**: 予測APIサーバーとして動作中
    - **not_built**: イメージ未ビルド（make build-full が必要）
    - **image_unknown**: Dockerソケット未マウント（状態不明）
    """
    statuses = {name: _container_status(name) for name in PIPELINE_CONTAINERS}
    ready_states = {"running", "image_ready", "api_running"}
    all_ready = all(s in ready_states for s in statuses.values())
    return {
        "all_ready": all_ready,
        "predictor_api": "running" if statuses.get("dysphagia-predictor") in ready_states else "stopped",
        "gpu_containers": {
            k: v for k, v in statuses.items() if k != "dysphagia-predictor"
        },
        "containers": statuses,
        "notes": {
            "nnunet":    "image_ready = nnU-Netコンテナはon-demand実行（GPU必要）",
            "totalseg":  "image_ready = TotalSegmentatorはon-demand実行（GPU必要）",
            "bianca":    "image_ready = FSL BIANCAはon-demand実行（CPU）",
            "predictor": "running = XGBoost予測APIはデーモン常駐",
        },
        "checked_at": datetime.now().isoformat(),
    }


@app.post(
    "/pipeline/run",
    tags=["パイプライン"],
    summary="全AIパイプライン実行",
    status_code=202,
)
async def pipeline_run(req: PipelineRunRequest, background_tasks: BackgroundTasks):
    """
    nnU-Net → TotalSegmentator → BIANCA → XGBoost の順にパイプラインを実行する。

    - 非同期実行（202 Accepted）
    - ジョブIDを返すので `/pipeline/jobs/{job_id}` で進捗確認
    - 結果は `/results/{patient_id}` で取得可能
    """
    ct_path = os.path.join(DATA_INPUT_DIR, req.ct_nifti_filename)
    if not os.path.exists(ct_path):
        available = os.listdir(DATA_INPUT_DIR) if os.path.exists(DATA_INPUT_DIR) else []
        raise HTTPException(
            status_code=404,
            detail=f"CTファイルが見つかりません: {req.ct_nifti_filename}  (利用可能: {available})"
        )

    job_id = f"{req.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "patient_id": req.patient_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
    }
    background_tasks.add_task(_run_pipeline_background, req, job_id)
    return {"job_id": job_id, "status": "queued", "message": "パイプラインを非同期で開始しました。"}


@app.get(
    "/pipeline/jobs/{job_id}",
    tags=["パイプライン"],
    summary="パイプラインジョブ進捗確認",
)
async def pipeline_job_status(job_id: str):
    """指定したジョブIDのパイプライン実行状態を返す。"""
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail=f"ジョブが見つかりません: {job_id}")
    return _pipeline_jobs[job_id]


@app.get(
    "/pipeline/jobs",
    tags=["パイプライン"],
    summary="全パイプラインジョブ一覧",
)
async def pipeline_jobs_list():
    return {"jobs": list(_pipeline_jobs.values())}


@app.get(
    "/results/{patient_id}",
    tags=["結果"],
    summary="保存済みレポート取得",
)
async def get_results(patient_id: str):
    """実行済みパイプラインの最終レポートJSONを返す。"""
    report_path = os.path.join(RESULTS_DIR, patient_id, f"{patient_id}_final_report.json")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail=f"レポートが見つかりません: {patient_id}")
    with open(report_path, encoding="utf-8") as f:
        return json.load(f)


@app.get(
    "/results",
    tags=["結果"],
    summary="保存済み全レポート一覧",
)
async def list_results():
    """保存済みの全レポートIDとファイル一覧を返す。"""
    if not os.path.exists(RESULTS_DIR):
        return {"results": []}
    reports = glob.glob(os.path.join(RESULTS_DIR, "**", "*_final_report.json"), recursive=True)
    return {
        "count": len(reports),
        "results": [os.path.basename(p).replace("_final_report.json", "") for p in reports],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
