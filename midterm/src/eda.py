"""Step 2: EDA on the full prepared data. Writes figures/*.pdf and results/eda.json."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, PREPARED, save_json

FRAUD_C, LEGIT_C = "#c0392b", "#2e86c1"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def main():
    df = pd.read_parquet(PREPARED)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fraud, legit = df[df.is_fraud == 1], df[df.is_fraud == 0]
    out = {}

    # 1. log amount by class
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bins = np.linspace(0, df.log_amt.max(), 60)
    ax.hist(legit.log_amt, bins=bins, density=True, alpha=0.6, color=LEGIT_C, label="legitimate")
    ax.hist(fraud.log_amt, bins=bins, density=True, alpha=0.6, color=FRAUD_C, label="fraud")
    ax.set_xlabel("log(1 + amount in USD)"); ax.set_ylabel("density"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "eda_log_amt.pdf"); plt.close(fig)
    out["median_amt"] = {"fraud": float(fraud.amt.median()), "legit": float(legit.amt.median())}
    out["mean_amt"] = {"fraud": float(fraud.amt.mean()), "legit": float(legit.amt.mean())}

    # 2. fraud rate by hour
    rate_hour = df.groupby("hour").is_fraud.mean() * 100
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    colors = [FRAUD_C if h in (22, 23, 0, 1, 2, 3) else "#7f8c8d" for h in rate_hour.index]
    ax.bar(rate_hour.index, rate_hour.values, color=colors)
    ax.set_xlabel("hour of day"); ax.set_ylabel("fraud rate (%)")
    fig.tight_layout(); fig.savefig(FIGURES / "eda_hour.pdf"); plt.close(fig)
    out["fraud_rate_by_hour_pct"] = rate_hour.round(3).to_dict()
    out["night_share"] = {"fraud": float(fraud.is_night.mean()), "legit": float(legit.is_night.mean())}

    # 3. fraud rate by category
    rate_cat = (df.groupby("category").is_fraud.mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.barh(rate_cat.index, rate_cat.values, color=FRAUD_C)
    ax.axvline(df.is_fraud.mean() * 100, color="k", lw=0.8, ls="--")
    ax.set_xlabel("fraud rate (%)  (dashed = overall)")
    fig.tight_layout(); fig.savefig(FIGURES / "eda_category.pdf"); plt.close(fig)
    out["fraud_rate_by_category_pct"] = rate_cat.round(3).to_dict()

    # 4. age density
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bins = np.linspace(14, 96, 50)
    ax.hist(legit.age, bins=bins, density=True, alpha=0.6, color=LEGIT_C, label="legitimate")
    ax.hist(fraud.age, bins=bins, density=True, alpha=0.6, color=FRAUD_C, label="fraud")
    ax.set_xlabel("cardholder age (years)"); ax.set_ylabel("density"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "eda_age.pdf"); plt.close(fig)
    out["mean_age"] = {"fraud": float(fraud.age.mean()), "legit": float(legit.age.mean())}

    # 5. distance boxplot
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.boxplot([legit.distance_km, fraud.distance_km], showfliers=False)
    ax.set_xticks([1, 2], ["legitimate", "fraud"])
    ax.set_ylabel("customer–merchant distance (km)")
    fig.tight_layout(); fig.savefig(FIGURES / "eda_distance.pdf"); plt.close(fig)
    out["mean_distance"] = {"fraud": float(fraud.distance_km.mean()), "legit": float(legit.distance_km.mean())}

    # before/after log transform
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2))
    axes[0].hist(df.amt.clip(upper=1500), bins=80, color="#7f8c8d")
    axes[0].set_xlabel("amount (USD, clipped at 1500 for display)")
    axes[1].hist(df.log_amt, bins=80, color="#7f8c8d")
    axes[1].set_xlabel("log(1 + amount)")
    fig.tight_layout(); fig.savefig(FIGURES / "prep_before_after.pdf"); plt.close(fig)

    out["fraud_rate_by_dow_pct"] = (df.groupby("day_of_week").is_fraud.mean() * 100).round(3).to_dict()
    save_json(out, "eda.json")
    print(out)


if __name__ == "__main__":
    main()
