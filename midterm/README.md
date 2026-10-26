# CSE558 Midterm Project — What separates a fraudulent card transaction from a normal one?

Group **Null & Void** (Darshan Bothra, Piyush Keshan, Virat Yadav). Statistical inference on the
Kaggle [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
(simulated with Sparkov; 1,852,394 transactions, 9,651 frauds).

## Setup

```bash
conda env create -f environment.yml   # from the repo root
conda activate data-science
```

Put `fraudTrain.csv` and `fraudTest.csv` in `dataset/` (download link in `dataset/README.md`).

## Run

```bash
cd midterm/src
python run_all.py          # or run the steps one by one, in this order:
python prepare.py          # load, quality checks, feature engineering -> results/prepared.parquet
python eda.py              # EDA figures -> figures/eda_*.pdf
python hypothesis_tests.py # pilot sample, power calculation, stratified sample, tests 1-5
python models.py           # logistic regression + decision tree, McNemar, ablation (~2-3 min)
```

Everything is seeded (`SEED = 42`, pilot seed 7), so reruns give the same numbers.

## What each step does

| Step | File | Output |
|---|---|---|
| Preparation | `src/prepare.py` | Missing values and duplicate checks, derived features (hour, night flag, day of week, age at transaction, haversine customer–merchant distance, log amount), PII and identifiers dropped. `results/data_summary.json` |
| EDA | `src/eda.py` | Five EDA plots plus a before/after plot of the log transform. `results/eda.json` |
| Hypothesis tests | `src/hypothesis_tests.py` | **Tests run on a sample, not the full data.** A pilot sample (500 per class) gives the standard deviations for a power calculation (α = 0.01 Bonferroni, power 0.8). The test sample is 2,000 fraud + 2,000 legitimate transactions drawn without replacement and disjoint from the pilot. Its ids are in `results/test_sample_trans_num.csv`. The script then runs five tests with alternative tests and effect sizes, repeats them on 5 more samples for robustness, and writes `results/hypothesis_tests.json` |
| Models | `src/models.py` | Train on `fraudTrain.csv`, choose the threshold on a 20% validation split of it, and evaluate once on `fraudTest.csv`. Compares linear and spline logistic regression with a decision tree, using McNemar's test and a paired bootstrap. Also runs a with/without ablation for each tested feature. `results/models.json`, `figures/model_pr_curves.pdf` |

### Hypothesis tests (on the 2,000 + 2,000 sample)

1. **Amount**: Welch t-test on log(1 + amt), one-sided. An F-test justifies Welch, and Mann–Whitney is the robustness check.
2. **Night (22:00–03:59)**: two-proportion z-test on P(night | class), one-sided, with an odds ratio. The χ² and Fisher tests are shown for comparison.
3. **Category**: χ² test of independence (14 × 2, df = 13), with Cramér's V and per-category odds ratios. The G-test is shown for comparison.
4. **Age**: Welch t-test, two-sided, with Mann–Whitney, KS, and TOST equivalence within ±3 years.
5. **Distance**: Welch t-test, two-sided, with Mann–Whitney, KS, and TOST equivalence within ±5 km.

Also included: a χ² goodness-of-fit test of fraud across days of the week.

### Model choices

All model choices were made on the validation split, never on the test file.
- **Class weighting:** `class_weight="balanced"` lowered validation PR-AUC for both models, so class imbalance is handled by tuning the decision threshold instead.
- **Logistic regression:** uses cubic-spline terms for amount, hour and age, because the effect of amount on fraud is not monotone.
- **Decision tree:** depth 12, at least 20 samples per leaf.

## Layout

```
midterm/
  src/        analysis code
  results/    JSON results, sample ids (prepared.parquet is regenerated, not committed)
  figures/    PDF figures used in the report
  report/     LaTeX source of the report
```
