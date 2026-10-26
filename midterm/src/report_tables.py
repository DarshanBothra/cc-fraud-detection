"""Write the model-comparison tables for the report straight from results/models.json."""
import json

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


def main():
    m = json.load(open(RESULTS / "models.json"))
    sel, fin, chosen = m["selection"], m["final"], m["chosen_model"]
    order = list(SHORT)

    # stage 1: imbalance strategy x model, validation PR-AUC (best per row in bold)
    rows = []
    for name in order:
        v = sel[name]["strategy_val_prauc"]
        best = sel[name]["strategy"]
        cells = [("\\textbf{%s}" % f3(v[s]) if s == best else f3(v[s])) for s in ["none", "class_weight", "undersample", "smote"]]
        rows.append(f"{SHORT[name]} & " + " & ".join(cells) + " \\\\")
    (MIDTERM / "report" / "tab_strategies.tex").write_text("\n".join(rows) + "\n")

    # stage 2: final models, ordered by validation PR-AUC
    rows = []
    for name in m["ranking_by_val_prauc"]:
        r = fin[name]
        cells = [SHORT[name], STRAT[r["strategy"]], f3(r["val_pr_auc"]), f"{r['precision']:.2f}",
                 f"{r['recall']:.2f}", f"{r['f1']:.2f}", f3(r["pr_auc"]), f"{r['roc_auc']:.3f}"]
        if name == chosen:
            cells = ["\\textbf{%s}" % c for c in cells]
        rows.append(" & ".join(cells) + " \\\\")
    (MIDTERM / "report" / "tab_models.tex").write_text("\n".join(rows) + "\n")

    # comparisons of the chosen model
    rows = []
    for c in m["comparisons"]:
        p = c["mcnemar_p"]
        ptxt = f"{p:.2g}" if p >= 1e-3 else f"$10^{{{int(round(__import__('math').log10(p)))}}}$" if p > 0 else "$<10^{-300}$"
        lo, hi = c["f1_diff_a_minus_b_ci95"]
        lo2, hi2 = c["prauc_diff_a_minus_b_ci95"]
        rows.append(f"{SHORT[c['b']]} & {c['a_right_b_wrong']} / {c['b_right_a_wrong']} & {ptxt} & "
                    f"${lo:+.3f}$ to ${hi:+.3f}$ & ${lo2:+.3f}$ to ${hi2:+.3f}$ \\\\")
    (MIDTERM / "report" / "tab_compare.tex").write_text("\n".join(rows) + "\n")
    print(open(MIDTERM / "report" / "tab_models.tex").read())
    print(open(MIDTERM / "report" / "tab_compare.tex").read())


if __name__ == "__main__":
    main()
