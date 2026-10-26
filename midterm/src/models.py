"""Step 4: binary classification (logistic regression and a decision tree).

Train on fraudTrain.csv rows, choose the threshold on a validation split carved from the training
rows, evaluate once on fraudTest.csv rows. Compare the models with McNemar's test and a paired
bootstrap CI, and run a with/without ablation for each feature group the hypothesis tests looked at.
Writes results/models.json and figures/model_*.pdf.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.contingency_tables import mcnemar

from common import FIGURES, PREPARED, SEED, save_json

NUMERIC = ["log_amt", "hour", "is_night", "day_of_week", "age", "gender_M", "log_city_pop", "distance_km"]
# feature groups tied to the five hypothesis tests, for the ablation
GROUPS = {
    "amount (test 1)": ["log_amt"],
    "time of day (test 2)": ["hour", "is_night"],
    "category (test 3)": "category",
    "age (test 4)": ["age"],
    "distance (test 5)": ["distance_km"],
}
N_BOOT = 200


def design(df):
    X = df[NUMERIC].copy()
    X = pd.concat([X, pd.get_dummies(df.category, prefix="cat", dtype=float)], axis=1)
    return X


def drop_group(X, group):
    cols = GROUPS[group]
    if cols == "category":
        return X.drop(columns=[c for c in X.columns if c.startswith("cat_")])
    return X.drop(columns=cols)


SPLINE_COLS = ["log_amt", "hour", "age"]


# Settings chosen on the validation split only (see README): class weighting lowered validation
# PR-AUC for both models, so imbalance is handled by tuning the decision threshold instead.
def make_lr_linear():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def make_lr():
    """Logistic regression with cubic-spline terms, so amount, hour and age can act non-linearly."""
    cols = lambda X: [c for c in SPLINE_COLS if c in X.columns]
    ct = ColumnTransformer([("spline", SplineTransformer(n_knots=8, degree=3), cols)],
                           remainder=StandardScaler())
    return make_pipeline(ct, LogisticRegression(max_iter=3000))


def make_tree():
    return DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED)


def best_threshold(y, score):
    prec, rec, thr = precision_recall_curve(y, score)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def fit_eval(model, Xtr, ytr, Xval, yval, Xte, yte):
    model.fit(Xtr, ytr)
    thr = best_threshold(yval, model.predict_proba(Xval)[:, 1])
    score = model.predict_proba(Xte)[:, 1]
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    metrics = {"threshold": thr, "precision": precision_score(yte, pred), "recall": recall_score(yte, pred),
               "f1": f1_score(yte, pred), "pr_auc": average_precision_score(yte, score),
               "roc_auc": roc_auc_score(yte, score), "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}
    return model, score, pred, metrics


def paired_bootstrap(y, a, b, fn, n_boot=N_BOOT, seed=SEED):
    """95% percentile CI for fn(y, a) - fn(y, b), resampling test rows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(fn(y[idx], a[idx]) - fn(y[idx], b[idx]))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main():
    df = pd.read_parquet(PREPARED)
    X, y = design(df), df.is_fraud.values
    tr_mask = (df.split == "train").values
    X_trainfull, y_trainfull = X[tr_mask], y[tr_mask]
    Xte, yte = X[~tr_mask], y[~tr_mask]
    Xtr, Xval, ytr, yval = train_test_split(X_trainfull, y_trainfull, test_size=0.2,
                                            stratify=y_trainfull, random_state=SEED)

    out = {"n_train": len(ytr), "n_val": len(yval), "n_test": len(yte),
           "frauds_test": int(yte.sum()), "features": list(X.columns)}

    _, _, _, out["logistic_regression_linear"] = fit_eval(make_lr_linear(), Xtr, ytr, Xval, yval, Xte, yte)
    lr, lr_score, lr_pred, out["logistic_regression"] = fit_eval(make_lr(), Xtr, ytr, Xval, yval, Xte, yte)
    dt, dt_score, dt_pred, out["decision_tree"] = fit_eval(make_tree(), Xtr, ytr, Xval, yval, Xte, yte)

    # all-legitimate baseline, to show why accuracy is not the headline
    out["baseline_all_legit_accuracy"] = float(1 - yte.mean())

    # interpretation: LR coefficients (standardised features) and tree importances
    names = lr[0].get_feature_names_out()
    coefs = lr[-1].coef_[0]
    out["lr_coefficients"] = dict(sorted(zip(names, coefs.round(4)), key=lambda kv: -abs(kv[1])))
    out["tree_importances"] = dict(sorted(zip(X.columns, dt.feature_importances_.round(4)),
                                          key=lambda kv: -kv[1]))

    # McNemar: do the two models make errors on different test rows?
    lr_ok, dt_ok = lr_pred == yte, dt_pred == yte
    table = [[int((lr_ok & dt_ok).sum()), int((lr_ok & ~dt_ok).sum())],
             [int((~lr_ok & dt_ok).sum()), int((~lr_ok & ~dt_ok).sum())]]
    mc = mcnemar(table, exact=False, correction=True)
    out["mcnemar_lr_vs_tree"] = {"table": table, "chi2": mc.statistic, "p": mc.pvalue}
    out["bootstrap_f1_diff_tree_minus_lr_ci95"] = paired_bootstrap(yte, dt_pred, lr_pred, f1_score)
    out["bootstrap_prauc_diff_tree_minus_lr_ci95"] = paired_bootstrap(yte, dt_score, lr_score, average_precision_score)

    # ablation: drop each tested feature group, refit, compare PR-AUC on the test set
    ablation = {}
    for name in GROUPS:
        row = {}
        for label, factory, full_score in (("logistic_regression", make_lr, lr_score),
                                           ("decision_tree", make_tree, dt_score)):
            m = factory().fit(drop_group(Xtr, name), ytr)
            s = m.predict_proba(drop_group(Xte, name))[:, 1]
            row[label] = {"pr_auc_without": average_precision_score(yte, s),
                          "pr_auc_drop": average_precision_score(yte, full_score) - average_precision_score(yte, s),
                          "drop_ci95": paired_bootstrap(yte, full_score, s, average_precision_score, n_boot=100)}
        ablation[name] = row
        print("ablation", name, {k: round(v["pr_auc_drop"], 4) for k, v in row.items()})
    out["ablation"] = ablation

    # figure: precision-recall curves
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for score, label, c in ((lr_score, "logistic regression", "#2e86c1"), (dt_score, "decision tree", "#c0392b")):
        p, r, _ = precision_recall_curve(yte, score)
        ax.plot(r, p, color=c, label=f"{label} (PR-AUC {average_precision_score(yte, score):.2f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "model_pr_curves.pdf"); plt.close(fig)

    save_json(out, "models.json")
    for k in ("logistic_regression_linear", "logistic_regression", "decision_tree"):
        print(k, {m: round(v, 4) if isinstance(v, float) else v for m, v in out[k].items()})
    print("mcnemar", out["mcnemar_lr_vs_tree"])
    print("f1 diff CI", out["bootstrap_f1_diff_tree_minus_lr_ci95"], "prauc diff CI", out["bootstrap_prauc_diff_tree_minus_lr_ci95"])
    print("top tree", list(out["tree_importances"].items())[:6])
    print("top lr", list(out["lr_coefficients"].items())[:6])


if __name__ == "__main__":
    main()
