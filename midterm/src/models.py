"""Step 4: binary classification. Every model and imbalance option listed in the plan, chosen on validation.

Data: train on fraudTrain.csv rows (split 80/20 into training and validation), test once on
fraudTest.csv rows. The test file is never used to choose anything.

Stage 1 (selection, on a stratified 25% subsample of the training split, scored on validation):
  for each model, try every imbalance strategy (none, class weights, undersampling, SMOTE) with
  default settings, keep the best, then try the model's small hyperparameter grid.
Stage 2: refit each model with its chosen setup on the full training split, pick the decision
  threshold on validation, evaluate on the test file. The final model is the one with the best
  validation PR-AUC.
Then: per-category fraud-rate feature check, McNemar's test and paired bootstrap against the
runner-up and the logistic-regression baseline, and a with/without ablation of each tested feature.
Writes results/models.json and figures/model_pr_curves.pdf.

LightGBM and XGBoost need the OpenMP runtime (libomp). The conda environment has it; in a plain
venv on macOS, set DYLD_LIBRARY_PATH to scikit-learn's bundled copy (see README).
"""
import inspect
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.contingency_tables import mcnemar

from common import FIGURES, PREPARED, SEED, save_json

warnings.filterwarnings("ignore")

NUMERIC = ["log_amt", "hour", "is_night", "day_of_week", "age", "gender_M", "log_city_pop", "distance_km"]
GROUPS = {  # feature groups tied to the five hypothesis tests, for the ablation
    "amount (test 1)": ["log_amt"],
    "time of day (test 2)": ["hour", "is_night"],
    "category (test 3)": "category",
    "age (test 4)": ["age"],
    "distance (test 5)": ["distance_km"],
}
STRATEGIES = ["none", "class_weight", "undersample", "smote"]
RESAMPLE_RATIO = 0.1          # undersampling / SMOTE target: 1 fraud per 10 legitimate
SELECTION_FRACTION = 0.25     # stage-1 subsample of the training split
N_BOOT = 200


# ---------------------------------------------------------------- models

def _spline_lr(n_knots):
    cols = lambda X: [c for c in ["log_amt", "hour", "age"] if c in X.columns]
    ct = ColumnTransformer([("spline", SplineTransformer(n_knots=n_knots, degree=3), cols)],
                           remainder=StandardScaler())
    return make_pipeline(ct, LogisticRegression(max_iter=3000))


# Boosting libraries are imported lazily so the rest of the module works without them.
def _lightgbm(p):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=500, num_leaves=63, learning_rate=p["learning_rate"],
                              random_state=SEED, verbose=-1)


def _xgboost(p):
    import xgboost as xgb
    return xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=p["learning_rate"],
                             tree_method="hist", random_state=SEED, n_jobs=-1, eval_metric="aucpr")


def _catboost(p):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=500, depth=8, learning_rate=p["learning_rate"],
                              random_seed=SEED, verbose=0, thread_count=-1)


# name: (builder(params), hyperparameter grid (first = default), max training rows or None)
MODELS = {
    "Logistic regression (linear)": (lambda p: make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=3000)),
                                     [{"C": 1.0}, {"C": 0.1}], None),
    "Logistic regression (splines)": (lambda p: _spline_lr(p["n_knots"]), [{"n_knots": 8}, {"n_knots": 5}], None),
    "Naive Bayes (Gaussian)": (lambda p: GaussianNB(var_smoothing=p["var_smoothing"]),
                               [{"var_smoothing": 1e-9}, {"var_smoothing": 1e-6}], None),
    "k-nearest neighbours": (lambda p: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=p["k"], weights="distance", n_jobs=-1)),
                             [{"k": 15}, {"k": 50}], 100_000),
    "Decision tree": (lambda p: DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=20, random_state=SEED),
                      [{"max_depth": 12}, {"max_depth": 8}, {"max_depth": 16}], None),
    "Random forest": (lambda p: RandomForestClassifier(n_estimators=200, min_samples_leaf=p["min_samples_leaf"], n_jobs=-1, random_state=SEED),
                      [{"min_samples_leaf": 1}, {"min_samples_leaf": 5}], None),
    "LightGBM": (_lightgbm, [{"learning_rate": 0.05}, {"learning_rate": 0.1}], None),
    "XGBoost": (_xgboost, [{"learning_rate": 0.05}, {"learning_rate": 0.1}], None),
    "CatBoost": (_catboost, [{"learning_rate": 0.05}, {"learning_rate": 0.1}], None),
    "Linear SVM": (lambda p: make_pipeline(StandardScaler(), LinearSVC(C=p["C"], dual=False, max_iter=5000)),
                   [{"C": 1.0}, {"C": 0.1}], None),
    "Neural network (MLP)": (lambda p: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=p["hidden"], early_stopping=True, max_iter=30, random_state=SEED)),
                             [{"hidden": (64, 32)}, {"hidden": (64,)}], None),
}
BASELINE = "Logistic regression (splines)"


# ---------------------------------------------------------------- helpers

def design(df):
    return pd.concat([df[NUMERIC], pd.get_dummies(df.category, prefix="cat", dtype=float)], axis=1)


def drop_group(X, group):
    cols = GROUPS[group]
    if cols == "category":
        return X.drop(columns=[c for c in X.columns if c.startswith("cat_")])
    return X.drop(columns=cols)


def resample(X, y, strategy):
    """Returns (X, y, sample_weight). Resampling touches only the data passed in (training rows)."""
    w = None
    if strategy == "undersample":
        X, y = RandomUnderSampler(sampling_strategy=RESAMPLE_RATIO, random_state=SEED).fit_resample(X, y)
    elif strategy == "smote":
        X, y = SMOTE(sampling_strategy=RESAMPLE_RATIO, random_state=SEED).fit_resample(X, y)
    elif strategy == "class_weight":
        pos = y.mean()
        w = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))
    return X, np.asarray(y), w


def cap_rows(X, y, cap):
    if cap is None or len(y) <= cap:
        return X, y
    X, _, y, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=SEED)
    return X, y


def final_step(est):
    return est.steps[-1][1] if isinstance(est, Pipeline) else est


def fit(name, params, X, y, strategy):
    build, _, cap = MODELS[name]
    X, y, w = resample(X, y, strategy)
    est = build(params)
    if w is None:
        est.fit(*cap_rows(X, y, cap))
    elif isinstance(est, Pipeline):
        est.fit(X, y, **{f"{est.steps[-1][0]}__sample_weight": w})
    else:
        est.fit(X, y, sample_weight=w)
    return est


def score(est, X):
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X)


def supports_weights(name):
    if name == "k-nearest neighbours":
        return False
    build, grid, _ = MODELS[name]
    return "sample_weight" in inspect.signature(final_step(build(grid[0])).fit).parameters


def best_threshold(y, s):
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.nanargmax(f1[:-1]))])


def test_metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return pred, {"threshold": thr, "precision": precision_score(y, pred), "recall": recall_score(y, pred),
                  "f1": f1_score(y, pred), "pr_auc": average_precision_score(y, s),
                  "roc_auc": roc_auc_score(y, s), "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def paired_bootstrap(y, a, b, fn, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(fn(y[i], a[i]) - fn(y[i], b[i]))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def compare(y, name_a, a_pred, a_score, name_b, b_pred, b_score):
    ok_a, ok_b = a_pred == y, b_pred == y
    table = [[int((ok_a & ok_b).sum()), int((ok_a & ~ok_b).sum())],
             [int((~ok_a & ok_b).sum()), int((~ok_a & ~ok_b).sum())]]
    mc = mcnemar(table, exact=False, correction=True)
    return {"a": name_a, "b": name_b, "table": table, "a_right_b_wrong": table[0][1],
            "b_right_a_wrong": table[1][0], "mcnemar_chi2": mc.statistic, "mcnemar_p": mc.pvalue,
            "f1_diff_a_minus_b_ci95": paired_bootstrap(y, a_pred, b_pred, f1_score),
            "prauc_diff_a_minus_b_ci95": paired_bootstrap(y, a_score, b_score, average_precision_score)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- main

def main():
    df = pd.read_parquet(PREPARED)
    X, y = design(df), df.is_fraud.values
    tr = (df.split == "train").values
    Xtr, Xval, ytr, yval = train_test_split(X[tr], y[tr], test_size=0.2, stratify=y[tr], random_state=SEED)
    Xte, yte = X[~tr], y[~tr]
    Xsel, _, ysel, _ = train_test_split(Xtr, ytr, train_size=SELECTION_FRACTION, stratify=ytr, random_state=SEED)
    out = {"n_train": len(ytr), "n_val": len(yval), "n_test": len(yte), "n_selection": len(ysel),
           "frauds_test": int(yte.sum()), "baseline_all_legit_accuracy": float(1 - yte.mean()),
           "features": list(X.columns), "strategies": STRATEGIES, "resample_ratio": RESAMPLE_RATIO}

    # stage 1: imbalance strategy, then hyperparameters, on the selection subsample
    selection = {}
    for name, (_, grid, _) in MODELS.items():
        by_strategy, by_params = {}, {}
        for strat in STRATEGIES:
            if strat == "class_weight" and not supports_weights(name):
                by_strategy[strat] = None
                continue
            t0 = time.time()
            est = fit(name, grid[0], Xsel, ysel, strat)
            by_strategy[strat] = average_precision_score(yval, score(est, Xval))
            log(f"[select] {name:32s} {strat:13s} val PR-AUC {by_strategy[strat]:.4f} ({time.time()-t0:.0f}s)")
        strategy = max((k for k, v in by_strategy.items() if v is not None), key=by_strategy.get)
        by_params[str(grid[0])] = by_strategy[strategy]
        for params in grid[1:]:
            est = fit(name, params, Xsel, ysel, strategy)
            by_params[str(params)] = average_precision_score(yval, score(est, Xval))
            log(f"[select] {name:32s} {str(params):26s} val PR-AUC {by_params[str(params)]:.4f}")
        selection[name] = {"strategy_val_prauc": by_strategy, "grid_val_prauc": by_params, "strategy": strategy,
                           "params": max(grid, key=lambda p: by_params[str(p)])}
    out["selection"] = {k: {**v, "params": str(v["params"])} for k, v in selection.items()}

    # stage 2: refit on the full training split, threshold on validation, evaluate on test
    final, scores_te, preds_te, fitted = {}, {}, {}, {}
    for name in MODELS:
        t0 = time.time()
        est = fit(name, selection[name]["params"], Xtr, ytr, selection[name]["strategy"])
        s_val = score(est, Xval)
        thr = best_threshold(yval, s_val)
        s_te = score(est, Xte)
        preds_te[name], m = test_metrics(yte, s_te, thr)
        m["val_pr_auc"] = average_precision_score(yval, s_val)
        m["strategy"], m["params"] = selection[name]["strategy"], str(selection[name]["params"])
        m["fit_seconds"] = round(time.time() - t0, 1)
        final[name], scores_te[name], fitted[name] = m, s_te, est
        log(f"[final]  {name:32s} val PR-AUC {m['val_pr_auc']:.4f}  test PR-AUC {m['pr_auc']:.4f}  F1 {m['f1']:.4f}")
    out["final"] = final
    ranking = sorted(final, key=lambda k: -final[k]["val_pr_auc"])
    best, runner_up = ranking[0], ranking[1]
    out["ranking_by_val_prauc"] = ranking
    out["chosen_model"] = best
    log("chosen:", best, "runner-up:", runner_up)

    # per-category fraud rate as an extra feature (target encoding from the training split only)
    rate = pd.Series(ytr, index=Xtr.index).groupby(df.loc[Xtr.index, "category"]).mean()
    add_rate = lambda Xp: Xp.assign(cat_fraud_rate=df.loc[Xp.index, "category"].map(rate).values)
    out["category_rate_feature"] = {}
    for name in dict.fromkeys([best, BASELINE]):
        est = fit(name, selection[name]["params"], add_rate(Xtr), ytr, selection[name]["strategy"])
        v = average_precision_score(yval, score(est, add_rate(Xval)))
        out["category_rate_feature"][name] = {"val_pr_auc_without": final[name]["val_pr_auc"], "val_pr_auc_with": v}
        log(f"[cat-rate] {name}: val PR-AUC {final[name]['val_pr_auc']:.4f} -> {v:.4f}")

    # statistical comparison on the test file: chosen model against the next five by
    # validation PR-AUC, the baseline and the decision tree
    others = [o for o in dict.fromkeys(ranking[1:6] + [BASELINE, "Decision tree"]) if o != best]
    out["comparisons"] = [compare(yte, best, preds_te[best], scores_te[best], o, preds_te[o], scores_te[o])
                          for o in others]
    for c in out["comparisons"]:
        log(f"[compare] {best} vs {c['b']}: McNemar p={c['mcnemar_p']:.3g}, "
            f"F1 diff CI {np.round(c['f1_diff_a_minus_b_ci95'], 3)}, PR-AUC diff CI {np.round(c['prauc_diff_a_minus_b_ci95'], 3)}")

    last = final_step(fitted[best])
    if hasattr(last, "feature_importances_"):
        imp = np.asarray(last.feature_importances_, dtype=float)
        out["chosen_feature_importances"] = dict(sorted(zip(X.columns, (imp / imp.sum()).round(4)), key=lambda kv: -kv[1]))

    # ablation: drop each tested feature group, refit, fall in test PR-AUC
    ablation = {}
    for group in GROUPS:
        row = {}
        for name in dict.fromkeys([best, BASELINE]):
            est = fit(name, selection[name]["params"], drop_group(Xtr, group), ytr, selection[name]["strategy"])
            s = score(est, drop_group(Xte, group))
            pr_auc = average_precision_score(yte, s)
            row[name] = {"pr_auc_without": pr_auc,
                         "pr_auc_drop": final[name]["pr_auc"] - pr_auc,
                         "drop_ci95": paired_bootstrap(yte, scores_te[name], s, average_precision_score, n_boot=100)}
        ablation[group] = row
        log("[ablation]", group, {k: round(v["pr_auc_drop"], 4) for k, v in row.items()})
    out["ablation"] = ablation

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    cmap = plt.get_cmap("tab20")
    for i, name in enumerate(ranking):
        p, r, _ = precision_recall_curve(yte, scores_te[name])
        ax.plot(r, p, lw=2.0 if name == best else 0.9, color=cmap(i), label=f"{name} ({final[name]['pr_auc']:.2f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend(frameon=False, fontsize=5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_pr_curves.pdf")
    plt.close(fig)

    save_json(out, "models.json")
    log("done")


if __name__ == "__main__":
    main()
