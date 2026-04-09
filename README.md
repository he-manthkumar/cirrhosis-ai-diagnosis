# Cirrhosis AI Diagnosis

Final year project focused on building an AI/ML-based system to assist in diagnosing liver cirrhosis using data-driven methods (primarily implemented in Jupyter Notebooks).

## Overview

This repository contains notebooks and supporting Python code for:
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering (if applicable)
- Model training and evaluation for cirrhosis diagnosis
- Result visualization and reporting

> Note: The project is largely notebook-based (Jupyter Notebook ~96% of the repository).

## Project Structure (Typical)

Depending on how your notebooks are organized, your repository may resemble:

- `*.ipynb` — Jupyter notebooks for experiments, training, evaluation
- `data/` — datasets (often excluded from git if large/private)
- `models/` — saved models/checkpoints (optional)
- `outputs/` / `results/` — plots, metrics, generated artifacts (optional)
- `requirements.txt` — Python dependencies (recommended)
- `README.md` — project documentation

If your folder names differ, update this section to match the actual structure.

## Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/he-manthkumar/cirrhosis-ai-diagnosis.git
cd cirrhosis-ai-diagnosis
```

### 2) Create and activate a virtual environment (recommended)

**Windows (PowerShell)**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

If you have a `requirements.txt`:
```bash
pip install -r requirements.txt
```

If not, a minimal setup for notebooks is:
```bash
pip install jupyter numpy pandas scikit-learn matplotlib seaborn
```

### 4) Run Jupyter Notebook / JupyterLab

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Open the main notebook(s) and run cells top-to-bottom.

## Dataset

Describe your dataset here:
- Source (Kaggle / UCI / hospital data / generated / etc.)
- Number of records and features
- Target label definition (e.g., cirrhosis positive/negative or stage classification)
- Any preprocessing steps (missing values, encoding, scaling)

If the dataset is not included in the repo, add clear instructions on how to obtain it and where to place it (e.g., `data/your_dataset.csv`).

## Models / Approach

Summarize what you implemented (edit as needed):
- Classical ML models: Logistic Regression, Random Forest, SVM, XGBoost, etc.
- (Optional) Deep learning models: ANN/CNN/RNN (if applicable)
- Cross-validation strategy
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

## Results

Add your best-performing model results here:
- Evaluation metrics
- Confusion matrix screenshot/plot (optional)
- ROC curve plot (optional)

Example:

- **Best model:** `<model name>`
- **ROC-AUC:** `<value>`
- **F1-score:** `<value>`

## How to Reproduce

1. Install dependencies
2. Download/prepare dataset and place it into `data/`
3. Run notebooks in this order:
   - `01_eda.ipynb`
   - `02_preprocessing.ipynb`
   - `03_training.ipynb`
   - `04_evaluation.ipynb`

(Replace filenames with your actual notebook names.)

## Future Improvements

- Improve feature engineering and model calibration
- Add hyperparameter tuning (GridSearch/Optuna)
- Add model interpretability (SHAP/LIME)
- Package the pipeline into a reusable Python module
- Build a simple UI (Streamlit/Flask) for demos

## Authors

- Hemanth Kumar (GitHub: @he-manthkumar)
- Rakesh Gopi
- Thawfiq Ahamed

