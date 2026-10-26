"""Step 2: EDA on the full prepared data. Writes figures/*.pdf and results/eda.json."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, NIGHT_HOURS, PREPARED, save_json

FRAUD_C, LEGIT_C = "#c0392b", "#2e86c1"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name)
    plt.close(fig)


def _class_density(legit, fraud, bins, xlabel, name):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.hist(legit, bins=bins, density=True, alpha=0.6, color=LEGIT_C, label="legitimate")
    ax.hist(fraud, bins=bins, density=True, alpha=0.6, color=FRAUD_C, label="fraud")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    _save(fig, name)


def main():
    df = pd.read_parquet(PREPARED)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fraud, legit = df[df.is_fraud == 1], df[df.is_fraud == 0]
    out = {}

    _class_density(legit.log_amt, fraud.log_amt, np.linspace(0, df.log_amt.max(), 60),
                   "log(1 + amount in USD)", "eda_log_amt.pdf")
    out["median_amt"] = {"fraud": float(fraud.amt.median()), "legit": float(legit.amt.median())}
    out["mean_amt"] = {"fraud": float(fraud.amt.mean()), "legit": float(legit.amt.mean())}

    rate_hour = df.groupby("hour").is_fraud.mean() * 100
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    colors = [FRAUD_C if h in NIGHT_HOURS else "#7f8c8d" for h in rate_hour.index]
    ax.bar(rate_hour.index, rate_hour.values, color=colors)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("fraud rate (%)")
    _save(fig, "eda_hour.pdf")
    out["fraud_rate_by_hour_pct"] = rate_hour.round(3).to_dict()
    out["night_share"] = {"fraud": float(fraud.is_night.mean()), "legit": float(legit.is_night.mean())}

    rate_cat = (df.groupby("category").is_fraud.mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.barh(rate_cat.index, rate_cat.values, color=FRAUD_C)
    ax.axvline(df.is_fraud.mean() * 100, color="k", lw=0.8, ls="--")
    ax.set_xlabel("fraud rate (%)  (dashed = overall)")
    _save(fig, "eda_category.pdf")
    out["fraud_rate_by_category_pct"] = rate_cat.round(3).to_dict()

    _class_density(legit.age, fraud.age, np.linspace(14, 96, 50),
                   "cardholder age (years)", "eda_age.pdf")
    out["mean_age"] = {"fraud": float(fraud.age.mean()), "legit": float(legit.age.mean())}

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.boxplot([legit.distance_km, fraud.distance_km], showfliers=False)
    ax.set_xticks([1, 2], ["legitimate", "fraud"])
    ax.set_ylabel("customer–merchant distance (km)")
    _save(fig, "eda_distance.pdf")
    out["mean_distance"] = {"fraud": float(fraud.distance_km.mean()), "legit": float(legit.distance_km.mean())}

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2))
    axes[0].hist(df.amt.clip(upper=1500), bins=80, color="#7f8c8d")
    axes[0].set_xlabel("amount (USD, clipped at 1500 for display)")
    axes[1].hist(df.log_amt, bins=80, color="#7f8c8d")
    axes[1].set_xlabel("log(1 + amount)")
    _save(fig, "prep_before_after.pdf")

    out["fraud_rate_by_dow_pct"] = (df.groupby("day_of_week").is_fraud.mean() * 100).round(3).to_dict()
    save_json(out, "eda.json")
    print(out)


if __name__ == "__main__":
    main()
