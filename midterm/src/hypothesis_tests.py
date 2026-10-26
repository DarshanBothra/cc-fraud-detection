"""Step 3: pilot sample -> power calculation -> stratified test sample -> five hypothesis tests.

The tests never touch the full data: they run on a stratified random sample (equal numbers
of fraud and legitimate transactions) whose size comes from a power calculation on a separate
pilot sample. Writes results/hypothesis_tests.json and results/test_sample_trans_num.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import PREPARED, RESULTS, SEED, save_json

ALPHA = 0.05
ALPHA_BONF = ALPHA / 5
POWER = 0.80
PILOT_PER_CLASS = 500
PILOT_SEED = 7
MIN_PER_CLASS = 2000

# smallest differences that matter (from the plan)
DELTA_LOG_AMT = np.log(1.5)
DELTA_NIGHT_PP = 0.10
DELTA_AGE = 3.0
DELTA_DIST = 5.0
OR_MATTERS = 2.0

z = stats.norm.ppf


def stratified_sample(df, per_class, seed, exclude=None):
    pool = df if exclude is None else df[~df.trans_num.isin(exclude)]
    parts = [g.sample(n=per_class, random_state=seed) for _, g in pool.groupby("is_fraud")]
    return pd.concat(parts, ignore_index=True)


def n_two_means(sd, delta, alpha, one_sided):
    za = z(1 - alpha) if one_sided else z(1 - alpha / 2)
    return 2 * (za + z(POWER)) ** 2 * sd ** 2 / delta ** 2


def n_two_props(p1, p2, alpha, one_sided):
    za = z(1 - alpha) if one_sided else z(1 - alpha / 2)
    pbar = (p1 + p2) / 2
    num = za * np.sqrt(2 * pbar * (1 - pbar)) + z(POWER) * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return num ** 2 / (p1 - p2) ** 2


def power_calculation(pilot):
    f, l = pilot[pilot.is_fraud == 1], pilot[pilot.is_fraud == 0]

    def means_entry(col, delta, one_sided):
        sd = np.sqrt((f[col].var() + l[col].var()) / 2)
        return {"sd_pilot": sd, "delta": delta,
                "n_per_group": int(np.ceil(n_two_means(sd, delta, ALPHA_BONF, one_sided)))}

    p_legit = l.is_night.mean()
    n_night = n_two_props(p_legit, min(p_legit + DELTA_NIGHT_PP, 0.999), ALPHA_BONF, True)
    req = {
        "test1_log_amt": means_entry("log_amt", DELTA_LOG_AMT, True),
        "test2_night": {"p_legit_pilot": p_legit, "delta": DELTA_NIGHT_PP, "n_per_group": int(np.ceil(n_night))},
        "test4_age": means_entry("age", DELTA_AGE, False),
        "test5_distance": means_entry("distance_km", DELTA_DIST, False),
    }
    return req, max(v["n_per_group"] for v in req.values())


def welch(x, y, alternative):
    res = stats.ttest_ind(x, y, equal_var=False, alternative=alternative)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    diff = x.mean() - y.mean()
    half = stats.t.ppf(0.975, df) * se
    return {"t": res.statistic, "df": df, "p": res.pvalue, "diff": diff,
            "ci95": [diff - half, diff + half], "se": se,
            "mean_fraud": x.mean(), "mean_legit": y.mean(), "sd_fraud": x.std(ddof=1), "sd_legit": y.std(ddof=1)}


def tost(w, delta):
    """Two one-sided tests for equivalence within +-delta, using the Welch SE and df."""
    t_low = (w["diff"] + delta) / w["se"]
    t_up = (w["diff"] - delta) / w["se"]
    p = max(1 - stats.t.cdf(t_low, w["df"]), stats.t.cdf(t_up, w["df"]))
    return {"delta": delta, "t_lower": t_low, "t_upper": t_up, "p": p}


def f_test(x, y):
    f = x.var(ddof=1) / y.var(ddof=1)
    d1, d2 = len(x) - 1, len(y) - 1
    p = 2 * min(stats.f.cdf(f, d1, d2), 1 - stats.f.cdf(f, d1, d2))
    return {"F": f, "df1": d1, "df2": d2, "p": p}


def odds_ratio(a, b, c, d):
    """a,b = exposed/unexposed among fraud; c,d = exposed/unexposed among legit (Woolf CI)."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return or_, [float(np.exp(np.log(or_) - 1.96 * se)), float(np.exp(np.log(or_) + 1.96 * se))]


def decision(p):
    return {"reject_at_0.05": bool(p < ALPHA), "reject_at_bonferroni_0.01": bool(p < ALPHA_BONF)}


def run_tests(s, full_legit_dow_share):
    f, l = s[s.is_fraud == 1], s[s.is_fraud == 0]
    out = {}

    # Test 1: log amount, one-sided
    w = welch(f.log_amt, l.log_amt, "greater")
    pooled = stats.ttest_ind(f.log_amt, l.log_amt, equal_var=True, alternative="greater")
    ratio_ci = [float(np.exp(w["ci95"][0])), float(np.exp(w["ci95"][1]))]
    out["test1_amount"] = {
        "f_test_variances": f_test(f.log_amt, l.log_amt), "welch": w, **decision(w["p"]),
        "ratio_geometric_means": float(np.exp(w["diff"])),
        "ratio_ci95": ratio_ci,
        "matters_threshold_ratio": 1.5,
        "matters": bool(ratio_ci[0] >= 1.5),
        "pooled_t": {"t": pooled.statistic, "df": len(f) + len(l) - 2, "p": pooled.pvalue},
        # same statistic as Welch, but referred to the normal distribution (sigma treated as known)
        "z_test": {"z": w["diff"] / w["se"], "p": float(stats.norm.sf(w["diff"] / w["se"]))},
        "median_amt": {"fraud": f.amt.median(), "legit": l.amt.median()},
    }

    # Test 2: night share, one-sided two-proportion z
    x1, n1, x2, n2 = f.is_night.sum(), len(f), l.is_night.sum(), len(l)
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    z_stat = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    p_z = 1 - stats.norm.cdf(z_stat)
    se_u = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff_ci = [p1 - p2 - 1.96 * se_u, p1 - p2 + 1.96 * se_u]
    or_, or_ci = odds_ratio(x1, n1 - x1, x2, n2 - x2)
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    chi2_2x2 = stats.chi2_contingency(table, correction=False)
    fisher = stats.fisher_exact(table, alternative="greater")
    out["test2_night"] = {
        "night_share_fraud": p1, "night_share_legit": p2, "z": z_stat, "p": p_z, **decision(p_z),
        "diff": p1 - p2, "diff_ci95": diff_ci,
        "odds_ratio": or_, "odds_ratio_ci95": or_ci,
        "matters_threshold_pp": DELTA_NIGHT_PP, "matters": bool(diff_ci[0] >= DELTA_NIGHT_PP),
        "chi2_2x2": {"chi2": chi2_2x2[0], "p_two_sided": chi2_2x2[1], "z_squared": z_stat ** 2},
        "fisher_exact": {"p": fisher.pvalue},
    }

    # Test 3: category x class, chi-square independence, with per-category ORs and
    # post-hoc two-proportion z-tests (Bonferroni over the number of categories)
    ct = pd.crosstab(s.category, s.is_fraud)
    chi2, p, dof, exp = stats.chi2_contingency(ct.values, correction=False)
    g, p_g, _, _ = stats.chi2_contingency(ct.values, correction=False, lambda_="log-likelihood")
    n_f, n_l = ct[1].sum(), ct[0].sum()
    per_cat = {}
    for cat in ct.index:
        a, c = ct.loc[cat, 1], ct.loc[cat, 0]
        o, ci = odds_ratio(a, n_f - a, c, n_l - c)
        pool = (a + c) / (n_f + n_l)
        zc = (a / n_f - c / n_l) / np.sqrt(pool * (1 - pool) * (1 / n_f + 1 / n_l))
        per_cat[cat] = {"fraud_count": int(a), "legit_count": int(c),
                        "share_in_fraud": a / n_f, "share_in_legit": c / n_l,
                        "odds_ratio": o, "or_ci95": ci, "matters": bool(ci[0] >= OR_MATTERS),
                        "posthoc_z": zc,
                        "posthoc_p_bonferroni": min(1.0, 2 * stats.norm.sf(abs(zc)) * len(ct.index))}
    out["test3_category"] = {
        "chi2": chi2, "df": dof, "p": p, **decision(p), "min_expected": float(exp.min()),
        "cramers_v": float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))),
        "g_test": {"G": g, "p": p_g}, "per_category": per_cat,
        "categories_that_matter": [k for k, v in per_cat.items() if v["matters"]],
        "posthoc_significant_bonferroni": [k for k, v in per_cat.items() if v["posthoc_p_bonferroni"] < ALPHA],
    }

    # Test 4: age, two-sided
    w = welch(f.age, l.age, "two-sided")
    ks = stats.ks_2samp(f.age, l.age)
    lo, hi = w["ci95"]
    out["test4_age"] = {
        "welch": w, **decision(w["p"]), "matters_threshold_years": DELTA_AGE,
        "matters": bool(min(abs(lo), abs(hi)) >= DELTA_AGE and np.sign(lo) == np.sign(hi)),
        "tost_3_years": tost(w, DELTA_AGE),
        "ks": {"D": ks.statistic, "p": ks.pvalue},
    }

    # Test 5: distance, two-sided, plus equivalence test
    w = welch(f.distance_km, l.distance_km, "two-sided")
    ks = stats.ks_2samp(f.distance_km, l.distance_km)
    out["test5_distance"] = {
        "welch": w, **decision(w["p"]), "matters_threshold_km": DELTA_DIST,
        "ci_inside_equivalence_band": bool(-DELTA_DIST < w["ci95"][0] and w["ci95"][1] < DELTA_DIST),
        "tost_5km": tost(w, DELTA_DIST),
        "ks": {"D": ks.statistic, "p": ks.pvalue},
    }

    # Extra: day-of-week goodness of fit, frauds vs legit proportions from the full data
    obs = f.day_of_week.value_counts().reindex(range(7), fill_value=0).values
    expct = full_legit_dow_share.reindex(range(7)).values * len(f)
    gof = stats.chisquare(obs, expct)
    out["extra_dow_gof"] = {"chi2": gof.statistic, "df": 6, "p": gof.pvalue,
                            "observed": obs, "expected": expct}
    return out


def representativeness(sample, df):
    rows = {}
    for cls in (0, 1):
        s, full = sample[sample.is_fraud == cls], df[df.is_fraud == cls]
        cat_diff = (s.category.value_counts(normalize=True)
                    - full.category.value_counts(normalize=True)).abs().max()
        rows["fraud" if cls else "legit"] = {
            "mean_amt": [s.amt.mean(), full.amt.mean()], "mean_age": [s.age.mean(), full.age.mean()],
            "mean_distance": [s.distance_km.mean(), full.distance_km.mean()],
            "night_share": [s.is_night.mean(), full.is_night.mean()],
            "max_abs_category_share_diff": cat_diff,
        }
    return rows


def main():
    df = pd.read_parquet(PREPARED)
    legit_dow = df[df.is_fraud == 0].day_of_week.value_counts(normalize=True)

    pilot = stratified_sample(df, PILOT_PER_CLASS, PILOT_SEED)
    pilot_ids = set(pilot.trans_num)
    req, needed = power_calculation(pilot)
    per_class = max(MIN_PER_CLASS, needed)

    sample = stratified_sample(df, per_class, SEED, exclude=pilot_ids)
    sample[["trans_num", "is_fraud"]].to_csv(RESULTS / "test_sample_trans_num.csv", index=False)

    results = {"pilot_per_class": PILOT_PER_CLASS, "power_calculation": req,
               "largest_required_per_class": needed, "sample_per_class": per_class,
               "alpha": ALPHA, "alpha_bonferroni": ALPHA_BONF,
               "representativeness": representativeness(sample, df),
               "tests": run_tests(sample, legit_dow)}

    # robustness: same tests on 5 more independent samples
    keys = ["test1_amount", "test2_night", "test3_category", "test4_age", "test5_distance"]
    agree = {k: [] for k in keys}
    for seed in range(1, 6):
        r = run_tests(stratified_sample(df, per_class, 1000 + seed, exclude=pilot_ids), legit_dow)
        for k in keys:
            agree[k].append(r[k]["reject_at_bonferroni_0.01"])
    results["robustness_5_samples_reject_at_0.01"] = agree

    save_json(results, "hypothesis_tests.json")
    t = results["tests"]
    t1, t2, t3, t4, t5 = (t[k] for k in keys)
    print("power:", {k: v["n_per_group"] for k, v in req.items()}, "-> per class", per_class)
    print("T1 ratio", t1["ratio_geometric_means"], t1["ratio_ci95"], t1["welch"]["p"])
    print("T2", t2["night_share_fraud"], t2["night_share_legit"], t2["diff_ci95"], t2["p"])
    print("T3 chi2", t3["chi2"], t3["p"], "V", t3["cramers_v"], t3["categories_that_matter"])
    print("T4", t4["welch"]["diff"], t4["welch"]["ci95"], t4["welch"]["p"], "tost", t4["tost_3_years"]["p"])
    print("T5", t5["welch"]["diff"], t5["welch"]["ci95"], t5["welch"]["p"], "tost", t5["tost_5km"]["p"])
    print("DOW", t["extra_dow_gof"]["p"])
    print("robust", agree)


if __name__ == "__main__":
    main()
