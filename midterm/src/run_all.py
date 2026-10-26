"""Run the whole midterm pipeline in order: prepare -> EDA -> hypothesis tests -> models."""
import eda
import hypothesis_tests
import models
import prepare

if __name__ == "__main__":
    for step in (prepare, eda, hypothesis_tests, models):
        print(f"\n=== {step.__name__} ===")
        step.main()
