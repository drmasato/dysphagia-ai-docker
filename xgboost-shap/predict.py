"""
XGBoost + SHAP 推論スクリプト
単体・バッチ予測とSHAP説明変数を出力する。
"""

import argparse
import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

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

RISK_THRESHOLDS = {
    "high":     0.30,
    "moderate": 0.70,
}

RISK_ACTIONS = {
    "high":     "VE/VF当日実施・ST緊急紹介・経口摂取禁止",
    "moderate": "RSST・MWST実施後にST紹介・嚥下リハ開始",
    "low":      "経過観察（3日後再評価）・食事中の観察継続",
}


def load_model(model_path: str):
    """保存済みXGBoostモデルを読み込む。"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
    return joblib.load(model_path)


def classify_risk(prob: float) -> str:
    if prob < RISK_THRESHOLDS["high"]:
        return "高リスク"
    elif prob < RISK_THRESHOLDS["moderate"]:
        return "中リスク"
    else:
        return "低リスク"


def classify_risk_en(prob: float) -> str:
    if prob < RISK_THRESHOLDS["high"]:
        return "high"
    elif prob < RISK_THRESHOLDS["moderate"]:
        return "moderate"
    else:
        return "low"


def predict_single(model, features: dict[str, Any]) -> dict:
    """単体症例の予測とSHAP説明を返す。"""
    df = pd.DataFrame([features])[FEATURE_COLUMNS].fillna(0)
    X = df.values

    prob = float(model.predict_proba(X)[0, 1])
    risk_ja = classify_risk(prob)
    risk_en = classify_risk_en(prob)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]
    shap_dict = {col: round(float(val), 4) for col, val in zip(FEATURE_COLUMNS, shap_values)}
    top_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return {
        "recovery_probability": round(prob, 4),
        "risk_level": risk_ja,
        "risk_level_en": risk_en,
        "recommended_action": RISK_ACTIONS[risk_en],
        "shap_values": shap_dict,
        "top_5_factors": [{"feature": k, "shap": v} for k, v in top_factors],
    }


def predict_batch(model, csv_path: str, output_dir: str) -> list[dict]:
    """バッチ予測を実行しCSVで保存する。"""
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS].fillna(0).values

    probs = model.predict_proba(X)[:, 1]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    results = []
    for i, (prob, sv) in enumerate(zip(probs, shap_values)):
        risk_en = classify_risk_en(float(prob))
        row = {
            "patient_id": df.get("patient_id", pd.Series(range(len(df))))[i],
            "recovery_probability": round(float(prob), 4),
            "risk_level": classify_risk(float(prob)),
            "recommended_action": RISK_ACTIONS[risk_en],
        }
        for col, val in zip(FEATURE_COLUMNS, sv):
            row[f"shap_{col}"] = round(float(val), 4)
        results.append(row)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "batch_predictions.csv")
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"バッチ予測完了: {out_path} ({len(results)}件)")
    return results


def main():
    parser = argparse.ArgumentParser(description="XGBoost 嚥下回復推論")
    parser.add_argument("--model", default="/workspace/models/dysphagia_xgb_model.joblib")
    subparsers = parser.add_subparsers(dest="mode")

    sp = subparsers.add_parser("single", help="単体予測")
    sp.add_argument("--features", required=True, help="特徴量JSON文字列またはファイルパス")
    sp.add_argument("--patient-id", default="unknown")

    bp = subparsers.add_parser("batch", help="バッチ予測")
    bp.add_argument("--data", required=True, help="入力CSVパス")
    bp.add_argument("--output-dir", default="/workspace/output")

    args = parser.parse_args()
    model = load_model(args.model)

    if args.mode == "single":
        if os.path.exists(args.features):
            with open(args.features) as f:
                features = json.load(f)
        else:
            features = json.loads(args.features)
        result = predict_single(model, features)
        result["patient_id"] = args.patient_id
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.mode == "batch":
        predict_batch(model, args.data, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
