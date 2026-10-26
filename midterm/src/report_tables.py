"""Write the model-comparison tables for the report straight from results/models.json."""
import json
import math

from common import MIDTERM, RESULTS

SHORT = {
    "Logistic regression (linear)": "Logistic regr.\\ (linear)",
    "Logistic regression (splines)": "Logistic regr.\\ (splines)",
    "Naive Bayes (Gaussian)": "Naive Bayes",
    "k-nearest neighbours": "$k$-nearest neighbours",
    "Decision tree": "Decision tree",
    "Random forest": "Random forest",
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "Linear SVM": "Linear SVM",
    "Neural network (MLP)": "Neural net (MLP)",
}
STRAT = {"none": "none", "class_weight": "class wt.", "undersample": "undersample", "smote": "SMOTE"}


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def bold(s):
    return "\\textbf{%s}" % s


def p_text(p):
    if p >= 1e-3:
        return f"{p:.2g}"
    if p > 0:
        return f"$10^{{{int(round(math.log10(p)))}}}$"
    return "$<10^{-300}$"


def main():
    m = json.loads((RESULTS / "models.json").read_text())
    report = MIDTERM / "report"

    # stage 1: imbalance strategy x model, validation PR-AUC (best per row in bold)
    rows = []
    for name in SHORT:
        v, best = m["selection"][name]["strategy_val_prauc"], m["selection"][name]["strategy"]
        cells = [bold(f3(v[s])) if s == best else f3(v[s]) for s in STRAT]
        rows.append(f"{SHORT[name]} & " + " & ".join(cells) + " \\\\")
    (report / "tab_strategies.tex").write_text("\n".join(rows) + "\n")

    # stage 2: final models, ordered by validation PR-AUC
    rows = []
    for name in m["ranking_by_val_prauc"]:
        r = m["final"][name]
        cells = [SHORT[name], STRAT[r["strategy"]], f3(r["val_pr_auc"]), f"{r['precision']:.2f}",
                 f"{r['recall']:.2f}", f"{r['f1']:.2f}", f3(r["pr_auc"]), f"{r['roc_auc']:.3f}"]
        if name == m["chosen_model"]:
            cells = [bold(c) for c in cells]
        rows.append(" & ".join(cells) + " \\\\")
    models_tex = "\n".join(rows) + "\n"
    (report / "tab_models.tex").write_text(models_tex)

    # comparisons of the chosen model
    rows = []
    for c in m["comparisons"]:
        lo, hi = c["f1_diff_a_minus_b_ci95"]
        lo2, hi2 = c["prauc_diff_a_minus_b_ci95"]
        rows.append(f"{SHORT[c['b']]} & {c['a_right_b_wrong']} / {c['b_right_a_wrong']} & {p_text(c['mcnemar_p'])} & "
                    f"${lo:+.3f}$ to ${hi:+.3f}$ & ${lo2:+.3f}$ to ${hi2:+.3f}$ \\\\")
    compare_tex = "\n".join(rows) + "\n"
    (report / "tab_compare.tex").write_text(compare_tex)

    print(models_tex)
    print(compare_tex)


if __name__ == "__main__":
    main()
