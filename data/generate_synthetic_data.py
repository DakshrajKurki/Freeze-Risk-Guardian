"""
generate_synthetic_data.py

Generates a synthetic dataset of merchant account "snapshots" and labels each
one with whether it would plausibly trigger a compliance-driven freeze/hold/
rolling-reserve in the next cycle.

WHY SYNTHETIC: Razorpay's real freeze/hold data is private. Rather than fake
the existence of a "real" dataset, we build the labeling logic directly from
documented, public freeze triggers (RBI/PMLA compliance guidance, Razorpay's
own published freeze-cause blog posts):
    - incomplete / stale KYC documentation
    - sudden volume spikes vs a merchant's declared baseline
    - elevated chargeback / refund ratios
    - transaction patterns inconsistent with declared business profile
      (e.g. unexpected international mix, geography mismatch)
    - merchant category risk tier

Be upfront about this in your README and pitch: the LABELS are a rule-based
simulation of policy, not ground truth from Razorpay's systems. What you are
proving is that a model CAN learn the underlying risk pattern with measured
precision/recall -- which is the actual ask of Track 02's "held-out test set"
bar. If you get access to any real anonymized data later, this pipeline drops
straight in.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000  # synthetic merchant snapshots


def generate(n=N):
    df = pd.DataFrame({
        "merchant_id": [f"M{100000+i}" for i in range(n)],
        "kyc_completeness_score": np.clip(RNG.normal(78, 18, n), 0, 100),
        "kyc_doc_age_days": np.clip(RNG.exponential(180, n), 0, 1500).astype(int),
        "days_since_onboarding": np.clip(RNG.exponential(400, n), 5, 3000).astype(int),
        "monthly_txn_volume_inr": np.clip(RNG.lognormal(12.5, 1.1, n), 5000, None),
        "baseline_3m_avg_volume_inr": None,  # filled below
        "chargeback_ratio_pct": np.clip(RNG.exponential(0.35, n), 0, 8),
        "refund_ratio_pct": np.clip(RNG.exponential(2.0, n), 0, 25),
        "international_txn_share_pct": np.clip(RNG.exponential(6, n), 0, 100),
        "avg_ticket_size_inr": np.clip(RNG.lognormal(6.5, 0.9, n), 50, None),
        "prior_risk_flags_count": RNG.poisson(0.4, n),
        "high_risk_category_flag": RNG.choice([0, 1], n, p=[0.85, 0.15]),
        "geo_mismatch_flag": RNG.choice([0, 1], n, p=[0.93, 0.07]),
    })

    # Baseline volume = current volume perturbed backwards, so we can compute
    # a genuine "spike ratio" rather than inventing it out of thin air.
    noise = RNG.normal(1.0, 0.18, n)
    df["baseline_3m_avg_volume_inr"] = np.clip(df["monthly_txn_volume_inr"] / np.clip(noise, 0.4, None), 5000, None)
    df["volume_spike_ratio"] = df["monthly_txn_volume_inr"] / df["baseline_3m_avg_volume_inr"]

    # --- Rule-based ground truth (weighted risk score -> probability -> label) ---
    risk_score = (
        0.028 * (100 - df["kyc_completeness_score"])
        + 0.004 * np.clip(df["kyc_doc_age_days"] - 365, 0, None)
        + 1.15 * np.clip(df["volume_spike_ratio"] - 1.8, 0, None)
        + 0.55 * df["chargeback_ratio_pct"]
        + 0.10 * df["refund_ratio_pct"]
        + 0.02 * df["international_txn_share_pct"]
        + 0.9 * df["prior_risk_flags_count"]
        + 1.1 * df["high_risk_category_flag"]
        + 1.6 * df["geo_mismatch_flag"]
        - 0.0015 * np.clip(df["days_since_onboarding"] - 180, 0, None)  # tenure lowers risk slightly
    )

    # Squash to probability, add labeling noise (real policy isn't a clean line)
    prob = 1 / (1 + np.exp(-(risk_score - 4.2)))
    prob = np.clip(prob + RNG.normal(0, 0.05, n), 0, 1)
    df["freeze_risk_label"] = (RNG.uniform(0, 1, n) < prob).astype(int)

    return df


if __name__ == "__main__":
    import os
    df = generate()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merchant_freeze_risk_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Positive (freeze-risk) rate: {df['freeze_risk_label'].mean():.2%}")
    print(df.head(3).to_string())
