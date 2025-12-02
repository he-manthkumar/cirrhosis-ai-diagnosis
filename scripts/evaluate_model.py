"""
Script to evaluate model metrics comprehensively.

Run this script after training to see detailed metrics:
    python -m scripts.evaluate_model
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

from backend.services.ml_service import MLService


def main():
    print("=" * 70)
    print("CIRRHOSIS AI DIAGNOSIS - MODEL EVALUATION")
    print("=" * 70)
    
    # Initialize ML service
    ml_service = MLService(model_dir="data/processed")
    
    # Load and prepare data
    print("\nLoading data...")
    df = pd.read_csv("data/raw/cirrhosis.csv")
    df = df.drop(columns=['ID', 'N_Days'], errors='ignore')
    df = df.dropna(subset=['Status'])
    
    # Prepare features
    X = df[ml_service.numerical_features + ml_service.categorical_features].copy()
    y = df['Status'].copy()
    
    # Create preprocessor and encode
    ml_service.preprocessor = ml_service._create_preprocessor()
    X_processed = ml_service.preprocessor.fit_transform(X)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    
    print(f"Dataset: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
    print(f"Classes: {list(classes)}")
    print(f"Class Distribution:")
    for cls, count in zip(*np.unique(y, return_counts=True)):
        print(f"  - {cls}: {count} ({count/len(y)*100:.1f}%)")
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate each base model
    print("\n" + "=" * 70)
    print("BASE MODEL PERFORMANCE (5-Fold Cross-Validation)")
    print("=" * 70)
    
    base_models = ml_service._create_base_models()
    model_results = {}
    
    for name, model in base_models.items():
        print(f"\n--- {name.upper()} ---")
        
        # Get cross-validated predictions
        y_pred = cross_val_predict(model, X_processed, y_encoded, cv=cv)
        y_proba = cross_val_predict(model, X_processed, y_encoded, cv=cv, method='predict_proba')
        
        # Calculate metrics
        accuracy = accuracy_score(y_encoded, y_pred)
        precision = precision_score(y_encoded, y_pred, average='weighted')
        recall = recall_score(y_encoded, y_pred, average='weighted')
        f1 = f1_score(y_encoded, y_pred, average='weighted')
        
        # ROC-AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(y_encoded, y_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = None
        
        model_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Train full stacked ensemble and evaluate
    print("\n" + "=" * 70)
    print("STACKED ENSEMBLE PERFORMANCE")
    print("=" * 70)
    
    # Generate meta-features using CV
    meta_features = np.zeros((X_processed.shape[0], len(base_models) * len(classes)))
    
    col_idx = 0
    for name, model in base_models.items():
        oof_proba = cross_val_predict(model, X_processed, y_encoded, cv=cv, method='predict_proba')
        meta_features[:, col_idx:col_idx + len(classes)] = oof_proba
        col_idx += len(classes)
    
    # Train and evaluate meta-model with CV
    from sklearn.linear_model import LogisticRegression
    meta_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    
    y_pred_ensemble = cross_val_predict(meta_model, meta_features, y_encoded, cv=cv)
    y_proba_ensemble = cross_val_predict(meta_model, meta_features, y_encoded, cv=cv, method='predict_proba')
    
    # Ensemble metrics
    accuracy = accuracy_score(y_encoded, y_pred_ensemble)
    precision = precision_score(y_encoded, y_pred_ensemble, average='weighted')
    recall = recall_score(y_encoded, y_pred_ensemble, average='weighted')
    f1 = f1_score(y_encoded, y_pred_ensemble, average='weighted')
    roc_auc = roc_auc_score(y_encoded, y_proba_ensemble, multi_class='ovr', average='weighted')
    
    print(f"\n  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Detailed classification report
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT (Ensemble)")
    print("-" * 70)
    print(classification_report(y_encoded, y_pred_ensemble, target_names=classes))
    
    # Confusion Matrix
    print("-" * 70)
    print("CONFUSION MATRIX (Ensemble)")
    print("-" * 70)
    cm = confusion_matrix(y_encoded, y_pred_ensemble)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df)
    
    # Model Comparison Summary
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    # Add ensemble to results
    model_results['ENSEMBLE'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    comparison_df = pd.DataFrame(model_results).T
    comparison_df = comparison_df.round(4)
    print(comparison_df.to_string())
    
    # Best model
    print(f"\n🏆 Best F1-Score: {comparison_df['f1'].idxmax()} ({comparison_df['f1'].max():.4f})")
    print(f"🏆 Best Accuracy: {comparison_df['accuracy'].idxmax()} ({comparison_df['accuracy'].max():.4f})")
    
    # Tree-Ensemble Agreement Rate
    print("\n" + "=" * 70)
    print("DECISION TREE - ENSEMBLE AGREEMENT")
    print("=" * 70)
    
    tree_preds = cross_val_predict(base_models['decision_tree'], X_processed, y_encoded, cv=cv)
    agreement_rate = (tree_preds == y_pred_ensemble).mean()
    print(f"  Agreement Rate: {agreement_rate:.2%}")
    print(f"  (How often the interpretable tree matches the ensemble)")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
