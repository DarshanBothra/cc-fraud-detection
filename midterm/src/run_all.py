"""Run the whole midterm pipeline in order: prepare -> EDA -> hypothesis tests -> models -> report tables."""
import eda
import hypothesis_tests
import models
import prepare
import report_tables

if __name__ == "__main__":
    for step in (prepare, eda, hypothesis_tests, models, report_tables):
        print(f"\n=== {step.__name__} ===")
        step.main()
