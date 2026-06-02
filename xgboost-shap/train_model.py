"""
XGBoost + SHAP 嚥下回復予測モデル学習スクリプト
5-fold CV + Optuna最適化 + SMOTE によるクラス不均衡対策を実装。
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    auc, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
TARGET_COLUMN = "dysphagia_recovery_90d"


def load_and_validate_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """データを読み込み、欠損値補完・バリデーションを行う。"""
    df = pd.read_csv(csv_path)
    print(f"データ読み込み: {len(df)}件")

    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"必須列が不足: {missing_cols}")

    for col in FEATURE_COLUMNS:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  欠損補完（中央値）: {col} = {median_val:.2f}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    print(f"クラス分布: {y.value_counts().to_dict()}")
    return X, y


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optunaのハイパーパラメータ探索目的関数。"""
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 100, 500),
        "max_depth":          trial.suggest_int("max_depth", 3, 8),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "use_label_encoder":  False,
        "eval_metric":        "logloss",
        "random_state":       42,
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        smote = SMOTE(random_state=42)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        clf = XGBClassifier(**params)
        clf.fit(X_tr_res, y_tr_res)
        y_prob = clf.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, y_prob))

    return float(np.mean(auc_scores))


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
    output_dir: str,
    n_splits: int = 5,
) -> dict:
    """最適パラメータで5-fold CVを実施し、最終モデルを学習する。"""
    X_arr = X.values
    y_arr = y.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []
    oof_probs = np.zeros(len(y_arr))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        smote = SMOTE(random_state=42)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        clf = XGBClassifier(**best_params)
        clf.fit(X_tr_res, y_tr_res)

        y_prob = clf.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = y_prob
        fold_auc = roc_auc_score(y_val, y_prob)
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold+1}: AUC = {fold_auc:.4f}")

    oof_auc = roc_auc_score(y_arr, oof_probs)
    print(f"\nOOF AUC: {oof_auc:.4f} ± {np.std(fold_aucs):.4f}")

    # 全データで最終モデルを学習
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_arr, y_arr)
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_res, y_res)

    model_path = os.path.join(output_dir, "dysphagia_xgb_model.joblib")
    joblib.dump(final_model, model_path)
    print(f"モデル保存: {model_path}")

    # SHAP値計算
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_arr)
    _plot_shap_summary(shap_values, X, output_dir)
    _plot_roc_curve(y_arr, oof_probs, output_dir)

    metrics = {
        "oof_auc": round(oof_auc, 4),
        "fold_aucs": [round(a, 4) for a in fold_aucs],
        "mean_auc": round(float(np.mean(fold_aucs)), 4),
        "std_auc": round(float(np.std(fold_aucs)), 4),
        "model_path": model_path,
        "feature_columns": FEATURE_COLUMNS,
    }
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def _plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP Summaryプロット保存完了")


def _plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (OOF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()
    print("ROC曲線プロット保存完了")


def main():
    parser = argparse.ArgumentParser(description="XGBoost 嚥下回復予測モデル学習")
    parser.add_argument("--data", required=True, help="学習CSVファイルパス")
    parser.add_argument("--output-dir", default="/workspace/models", help="モデル出力ディレクトリ")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna試行回数")
    parser.add_argument("--n-splits", type=int, default=5, help="CV分割数")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X, y = load_and_validate_data(args.data)

    print(f"\n[Optuna] ハイパーパラメータ最適化 ({args.n_trials}試行)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X.values, y.values),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    best_params = {
        **study.best_params,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    print(f"最良パラメータ: {best_params}")
    print(f"最良AUC（探索時）: {study.best_value:.4f}")

    params_path = os.path.join(args.output_dir, "best_params.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\n[5-fold CV] 最終モデル学習 ({args.n_splits}分割)...")
    metrics = train_final_model(X, y, best_params, args.output_dir, args.n_splits)

    print("\n=== 学習完了 ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
