# 💳 CreditSense AI — Explainable Credit Scoring

A portfolio-grade creditworthiness prediction system aligned with CodeAlpha Task 1.

### Highlights
- Leakage-aware preprocessing
- Logistic Regression + Random Forest + HistGradientBoosting
- 5-fold stratified cross-validation
- Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC
- Probability calibration
- Threshold analysis
- Confusion matrix + ROC/PR curves
- Feature-importance report
- Applicant-level prediction explanation
- Streamlit executive dashboard
- Model card + responsible-use warning

### Dataset
The training script uses the OpenML `credit-g` dataset. Dataset access is performed at training time; no fabricated metrics are included.

### Run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

> This is an educational ML project. It is not a lending decision system and should not be used to approve/deny real credit.
