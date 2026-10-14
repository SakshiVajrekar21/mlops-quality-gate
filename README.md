# Automated ML Quality Gate and Artifact Delivery

## Project overview
This repository trains and validates a binary classifier, compares it with a baseline, runs application tests, and uploads a downloadable model package only when all required checks pass.

## Dataset and task
- **Dataset:** Breast Cancer Wisconsin Diagnostic
- **Source:** [scikit-learn public dataset CSV](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/data/breast_cancer.csv)
- **Download URL used by the script:** `https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/breast_cancer.csv`
- **Target variable:** `diagnosis`, converted to `1 = malignant` and `0 = benign`
- **Input features:** 30 numeric diagnostic measurements, represented in this project as `feature_01` through `feature_30`
- **Prediction task:** Binary classification of malignant versus benign
- The CSV is downloaded automatically; no login is required.

## Evaluation metric
The metric is **F1-score for the malignant class**. F1 balances precision and recall, so it is more informative than accuracy when the two classes have different frequencies and the positive class is important. This is a classroom demonstration, not a medical diagnostic tool.

## Baseline and quality gate
- **Baseline:** `DummyClassifier(strategy="most_frequent")`
- **Candidate:** `StandardScaler` + `LogisticRegression`
- **Split:** 80% training / 20% validation, stratified, `random_state=42`
- **Margin:** 0.05 F1 points
- **Gate:** candidate F1 must be greater than or equal to baseline F1 + 0.05.

The margin requires a meaningful improvement rather than accepting a tiny difference that may be due to noise. If too low, a barely useful improvement could pass. If too high, a useful model could fail even when it clearly beats the baseline. The metric, split and margin must remain unchanged during demonstrations.

## Setup and local run
Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then run:

```bash
python -m pip install -r requirements.txt
python train.py
pytest -q
```

Training creates `artifacts/model.joblib` and `artifacts/metrics.json`. Tests run after training and load the saved model from the same workspace.

## Application tests
Automated tests verify that:
1. The saved model package can be loaded.
2. A valid 30-feature sample returns a dictionary with an integer prediction and a class label.
3. A sample missing a required feature is rejected with a clear `ValueError`.

## GitHub Actions workflow
The workflow at `.github/workflows/mlops-pipeline.yml` runs on pushes to `main` and can also be started manually. Training, quality-gate evaluation, and tests run in one job. The artifact upload step only runs after the earlier steps succeed.

The downloadable artifact contains:
- fitted model and preprocessing pipeline
- metrics report
- prediction script
- dependency file
- workflow run number and commit identifier

## Failure and recovery evidence
After completing the demonstrations, replace the placeholders below with actual GitHub Actions run URLs. Do not invent or reuse URLs.

- **Failure A — quality gate:** [Add failed workflow run URL]
  - Temporary change: replace the Logistic Regression candidate in `train.py` with `DummyClassifier(strategy="most_frequent")`.
  - Expected blocker: quality-gate check exits non-zero because the candidate does not improve on the baseline by 0.05.
  - Expected result: tests and artifact upload are not reached; no package is published.
  - Restore the Logistic Regression pipeline before Failure B.

- **Failure B — application test:** [Add failed workflow run URL]
  - Temporary change: in `predict.py`, change the missing-feature error text so it no longer contains `missing required feature`.
  - Expected blocker: `test_missing_feature_is_rejected_clearly` fails its message match.
  - Expected result: the job fails before artifact upload; no package is published.
  - Restore the original error text before the final successful run.

- **Final successful run:** [Add successful workflow run URL]
  - **Artifact:** `validated-model-package-run-<run-number>`; download it from the successful run's **Artifacts** section.
  - Download a local copy before the GitHub Actions artifact retention period expires.

## Brief MLOps maturity assessment
This is a basic, reproducible CI and artifact-delivery workflow: code changes trigger automated data validation, training, baseline comparison, quality gating, application tests, and version-identified package publication. It is best described as an **early / Level 1-style automated ML pipeline** for this lab, rather than a fully mature production MLOps platform. It does not include scheduled retraining, experiment/model registry, production deployment, live monitoring, drift detection, approval controls, or automated rollback. Those capabilities would be needed to progress toward a more mature production setup.

## Submission checklist
- [ ] Push all code and `.github/workflows/mlops-pipeline.yml` to GitHub.
- [ ] Confirm Failure A fails at the quality gate and publishes no artifact.
- [ ] Restore the model and confirm Failure B fails at the application test and publishes no artifact.
- [ ] Restore the test code and confirm the final run succeeds.
- [ ] Replace all three README placeholders with real run URLs.
- [ ] Download the successful artifact and keep a local backup.
- [ ] Submit the repository link through Classroom and ensure the instructor has access.
