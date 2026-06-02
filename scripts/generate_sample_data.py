"""
合成学習データ生成スクリプト
200症例の脳卒中後嚥下障害データを生成する（学習・テスト用）。
"""

import argparse
import os

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_dataset(n_samples: int = 200, output_path: str = None) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    age = rng.integers(45, 90, n_samples).astype(float)
    sex_male = rng.integers(0, 2, n_samples).astype(float)

    # 臨床指標（現実的な分布を模倣）
    nihss_score   = rng.gamma(shape=2.5, scale=3.0, size=n_samples).clip(0, 42)
    aspects_score = rng.choice(range(0, 11), n_samples, p=[0.02,0.03,0.05,0.08,0.12,0.15,0.18,0.16,0.12,0.06,0.03])
    aspects_score = aspects_score.astype(float)

    lesion_volume_ml = rng.lognormal(mean=1.8, sigma=1.2, size=n_samples).clip(0.1, 200)
    wmh_volume_ml    = rng.lognormal(mean=0.5, sigma=1.3, size=n_samples).clip(0, 50)

    # 嚥下筋（年齢・性別依存）
    tongue_base = np.where(sex_male == 1, 4.5, 3.8)
    tongue_csa_cm2 = (tongue_base - age * 0.02 + rng.normal(0, 0.5, n_samples)).clip(1.5, 7.0)

    spc_base = np.where(sex_male == 1, 2.0, 1.6)
    spc_volume = (spc_base - age * 0.01 + rng.normal(0, 0.3, n_samples)).clip(0.5, 4.0)

    # 嚥下ホットスポット（梗塞体積と相関）
    hotspot_affected_count = np.minimum(
        rng.poisson(lam=lesion_volume_ml / 20, size=n_samples), 4
    ).astype(float)

    albumin_gdl = rng.normal(3.8, 0.5, n_samples).clip(2.0, 5.0)
    bmi         = rng.normal(22.0, 3.5, n_samples).clip(14.0, 40.0)

    prior_stroke       = rng.binomial(1, 0.2, n_samples).astype(float)
    atrial_fibrillation = rng.binomial(1, 0.25, n_samples).astype(float)
    diabetes            = rng.binomial(1, 0.3, n_samples).astype(float)

    # 回復確率の計算（ロジスティックモデルで模倣）
    logit = (
        0.15 * aspects_score
        - 0.08 * nihss_score
        - 0.03 * lesion_volume_ml
        - 0.05 * wmh_volume_ml
        + 0.20 * tongue_csa_cm2
        + 0.15 * spc_volume
        - 0.03 * age
        + 0.10 * albumin_gdl
        - 0.20 * atrial_fibrillation
        - 0.15 * diabetes
        + 1.5
        + rng.normal(0, 0.5, n_samples)
    )
    prob = 1 / (1 + np.exp(-logit))
    recovery = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "patient_id":       [f"PT-{i:04d}" for i in range(1, n_samples + 1)],
        "age":              age,
        "sex_male":         sex_male,
        "nihss_score":      nihss_score.round(1),
        "aspects_score":    aspects_score,
        "lesion_volume_ml": lesion_volume_ml.round(2),
        "wmh_volume_ml":    wmh_volume_ml.round(2),
        "tongue_csa_cm2":   tongue_csa_cm2.round(2),
        "superior_pharyngeal_constrictor_volume_cm3": spc_volume.round(2),
        "hotspot_affected_count": hotspot_affected_count,
        "albumin_gdl":      albumin_gdl.round(2),
        "bmi":              bmi.round(1),
        "prior_stroke":     prior_stroke,
        "atrial_fibrillation": atrial_fibrillation,
        "diabetes":         diabetes,
        "dysphagia_recovery_90d": recovery,
        "recovery_probability_true": prob.round(4),
    })

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"合成データ生成完了: {output_path} ({n_samples}件)")
        print(f"  回復あり: {recovery.sum()}件 ({recovery.mean()*100:.1f}%)")
        print(f"  回復なし: {(1-recovery).sum()}件 ({(1-recovery.mean())*100:.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(description="合成学習データ生成")
    parser.add_argument("--n-samples", type=int, default=200, help="生成症例数")
    parser.add_argument("--output", default="/workspace/data/training_data.csv", help="出力CSVパス")
    args = parser.parse_args()

    generate_dataset(n_samples=args.n_samples, output_path=args.output)


if __name__ == "__main__":
    main()
