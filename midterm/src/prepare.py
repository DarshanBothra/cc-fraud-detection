"""Step 1: load both CSVs, check quality, engineer features, save a prepared table.

Writes results/prepared.parquet and results/data_summary.json.
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import DATASET, NIGHT_HOURS, PREPARED, haversine_km, save_json

PII_AND_IDS = ["cc_num", "first", "last", "street", "trans_num", "merchant",
               "job", "city", "zip", "state", "dob", "trans_date_trans_time"]


def iqr_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())


def main():
    train = pd.read_csv(DATASET / "fraudTrain.csv", index_col=0)
    test = pd.read_csv(DATASET / "fraudTest.csv", index_col=0)
    train["split"] = "train"
    test["split"] = "test"
    df = pd.concat([train, test], ignore_index=True)

    raw_rows = len(df)
    raw_cols = len(pd.read_csv(DATASET / "fraudTrain.csv", nrows=0).columns)  # includes unnamed index
    missing = df.isna().sum()
    summary = {
        "rows_train": len(train), "rows_test": len(test), "rows_total": raw_rows,
        "raw_columns_incl_index": raw_cols,
        "frauds_train": int(train.is_fraud.sum()), "frauds_test": int(test.is_fraud.sum()),
        "frauds_total": int(df.is_fraud.sum()), "fraud_rate": float(df.is_fraud.mean()),
        "missing_values_total": int(missing.sum()),
        "missing_by_column": {k: int(v) for k, v in missing.items() if v},
        "duplicate_trans_num": int(df.trans_num.duplicated().sum()),
        "duplicate_rows": int(df.drop(columns=["split"]).duplicated().sum()),
        "n_categories": int(df.category.nunique()),
        "n_customers": int(df.cc_num.nunique()), "n_merchants": int(df.merchant.nunique()),
    }

    ts = pd.to_datetime(df.trans_date_trans_time)
    dob = pd.to_datetime(df.dob)
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.day_of_week
    df["month"] = ts.dt.month
    df["date"] = ts.dt.date.astype(str)
    df["is_night"] = df.hour.isin(NIGHT_HOURS).astype(int)
    df["age"] = (ts - dob).dt.days / 365.25
    df["distance_km"] = haversine_km(df.lat, df.long, df.merch_lat, df.merch_long)
    df["log_amt"] = np.log1p(df.amt)
    df["log_city_pop"] = np.log10(df.city_pop)
    df["gender_M"] = (df.gender == "M").astype(int)

    summary["date_range"] = [str(ts.min()), str(ts.max())]
    summary["before_after"] = {
        "amt_skew": float(stats.skew(df.amt)), "log_amt_skew": float(stats.skew(df.log_amt)),
        "amt_iqr_outliers": iqr_outliers(df.amt.values),
        "log_amt_iqr_outliers": iqr_outliers(df.log_amt.values),
        "amt_median": float(df.amt.median()), "amt_max": float(df.amt.max()),
        "age_min": float(df.age.min()), "age_max": float(df.age.max()),
        "distance_min": float(df.distance_km.min()), "distance_max": float(df.distance_km.max()),
    }

    # 3M rule: encoded design = numeric/derived columns + one-hot category + target
    numeric = ["amt", "gender_M", "city_pop", "lat", "long", "merch_lat", "merch_long", "age",
               "hour", "day_of_week", "month", "distance_km", "unix_time", "is_fraud"]
    encoded_cols = len(numeric) + summary["n_categories"]
    summary["encoded_columns"] = encoded_cols
    summary["cells_raw"] = raw_rows * raw_cols
    summary["cells_encoded"] = raw_rows * encoded_cols
    summary["dropped_columns"] = PII_AND_IDS

    keep = ["trans_num", "split", "date", "category", "amt", "log_amt", "hour", "is_night",
            "day_of_week", "month", "age", "gender_M", "city_pop", "log_city_pop",
            "distance_km", "is_fraud"]
    df[keep].to_parquet(PREPARED, index=False)
    save_json(summary, "data_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
