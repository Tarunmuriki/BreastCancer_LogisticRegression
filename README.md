# Breast Cancer Logistic Regression

## Objective
Build a binary classifier using Logistic Regression on the Breast Cancer Wisconsin dataset. The model predicts whether a tumor is malignant (0) or benign (1).

## Project Structure
- `src/cancer_prediction_model.py`
- `outputs/confusion_matrix.png`
- `outputs/roc_auc_curve.png`
- `requirements.txt`
- `README.md`
- `.gitignore`

## Tools Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

## Steps
1. Load the dataset
2. Split train and test data
3. Standardize features using `StandardScaler`
4. Train a Logistic Regression model
5. Evaluate using:
   - Confusion Matrix
   - Precision
   - Recall
   - ROC-AUC
6. Tune the classification threshold

## How to Run
```bash
python3 src/cancer_prediction_model.py
```

## Expected Results
- Accuracy: around `95% - 99%`
- Precision: `0.95+`
- Recall: `0.95+`
- ROC AUC: `0.99`

## Author
Your Name
