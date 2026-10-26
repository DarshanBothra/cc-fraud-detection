# CSE558 Midterm Project — What separates a fraudulent card transaction from a normal one?

Group **Null & Void** (Darshan Bothra, Piyush Keshan). Statistical inference on the
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
python models.py           # 11 models x 4 imbalance options, selection, McNemar, ablation (~20 min)
python report_tables.py    # writes the model tables that report/report.tex reads
```

Everything is seeded (`SEED = 42`, pilot seed 7), so reruns give the same numbers.

## What each step does

| Step | File | Output |
|---|---|---|
| Preparation | `src/prepare.py` | Missing values and duplicate checks, derived features (hour, night flag, day of week, age at transaction, haversine customer–merchant distance, log amount), PII and identifiers dropped. `results/data_summary.json` |
| EDA | `src/eda.py` | Five EDA plots plus a before/after plot of the log transform. `results/eda.json` |
| Hypothesis tests | `src/hypothesis_tests.py` | **Tests run on a sample, not the full data.** A pilot sample (500 per class) gives the standard deviations for a power calculation (α = 0.01 Bonferroni, power 0.8). The test sample is 2,000 fraud + 2,000 legitimate transactions drawn without replacement and disjoint from the pilot. Its ids are in `results/test_sample_trans_num.csv`. The script then runs five tests with alternative tests and effect sizes, repeats them on 5 more samples for robustness, and writes `results/hypothesis_tests.json` |
| Models | `src/models.py` | Every model and class-imbalance option in the plan, compared fairly. Models: linear and spline logistic regression, Gaussian Naive Bayes, k-nearest neighbours, decision tree, random forest, LightGBM, XGBoost, CatBoost, linear SVM and a neural network (MLP). Imbalance options: none, class weights, random undersampling and SMOTE (both resample to 1 fraud per 10 legitimate, training rows only). Training uses `fraudTrain.csv`, split 80/20 into training and validation. Stage 1 picks each model's imbalance option, then its hyperparameters, on a 25% subsample, scored by validation PR-AUC. Stage 2 refits on the full training split, sets the threshold on validation and evaluates once on `fraudTest.csv`. The final model is the best by **validation** PR-AUC. The step also checks a per-category fraud-rate feature, runs McNemar's test and a paired bootstrap against the next five models and the logistic-regression baseline, and does a with/without ablation of each tested feature. Writes `results/models.json` and `figures/model_pr_curves.pdf`. Takes about 20 minutes. |

### Hypothesis tests (on the 2,000 + 2,000 sample)

1. **Amount**: Welch t-test on log(1 + amt), one-sided. An F-test justifies Welch; the pooled t-test and z-test are also run for comparison.
2. **Night (22:00–03:59)**: two-proportion z-test on P(night | class), one-sided, with an odds ratio. The χ² and Fisher tests are shown for comparison.
3. **Category**: χ² test of independence (14 × 2, df = 13), with Cramér's V and per-category odds ratios. The G-test is shown for comparison, plus post-hoc per-category two-proportion tests with Bonferroni correction.
4. **Age**: Welch t-test, two-sided, with KS and TOST equivalence within ±3 years.
5. **Distance**: Welch t-test, two-sided, with KS and TOST equivalence within ±5 km.

Also included: a χ² goodness-of-fit test of fraud across days of the week.

### Running the models on macOS without conda

LightGBM and XGBoost need the OpenMP runtime (`libomp`). The conda environment includes it (`llvm-openmp`). In a plain virtualenv on macOS you can reuse scikit-learn's bundled copy:

```bash
DYLD_LIBRARY_PATH=$(python -c "import sklearn,os;print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))") python models.py
```

## Layout

```
midterm/
  src/        analysis code
  results/    JSON results, sample ids (prepared.parquet is regenerated, not committed)
  figures/    PDF figures used in the report
  report/     LaTeX source of the report
```
