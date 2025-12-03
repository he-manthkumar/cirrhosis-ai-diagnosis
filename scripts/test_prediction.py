"""
Script to test prediction and explanation workflow.

Run this script to test the full prediction + explanation pipeline:
    python -m scripts.test_prediction
"""
import sys
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.models.patient import PatientInput
from backend.services.ml_service import MLService
from backend.services.explanation_service import ExplanationService


def evaluate_on_test_set(ml_service):
    """Evaluate the model on the held-out test set and report metrics."""
    print("\n" + "=" * 70)
    print("📊 EVALUATION ON HELD-OUT TEST SET (20% - Never seen during training)")
    print("=" * 70)
    
    # Load the held-out test set
    test_data_path = project_root / "backend" / "models" / "test_data.pkl"
    if not test_data_path.exists():
        print("❌ Test data not found! Run training first.")
        return
    
    test_data = joblib.load(test_data_path)
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    classes = np.array(ml_service.classes_)
    print(f"\n✓ Test set size: {len(y_test)} samples")
    print(f"✓ Classes: {list(classes)}")
    
    # Get predictions from each base model
    base_models = ml_service.base_models
    
    print("\n" + "-" * 70)
    print("BASE MODEL PERFORMANCE")
    print("-" * 70)
    
    model_results = {}
    all_probas = {}
    
    for name, model in base_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        all_probas[name] = y_proba
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        
        model_results[name] = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc
        }
        
        print(f"\n{name.upper()}:")
        print(f"  Accuracy:  {acc:.4f} ({acc:.2%})")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc:.4f}")
    
    # Ensemble prediction
    print("\n" + "-" * 70)
    print("STACKED ENSEMBLE PERFORMANCE")
    print("-" * 70)
    
    # Create meta-features for ensemble
    meta_features = np.hstack([all_probas[name] for name in base_models.keys()])
    y_pred_ensemble = ml_service.meta_model.predict(meta_features)
    y_proba_ensemble = ml_service.meta_model.predict_proba(meta_features)
    
    acc = accuracy_score(y_test, y_pred_ensemble)
    prec = precision_score(y_test, y_pred_ensemble, average='weighted')
    rec = recall_score(y_test, y_pred_ensemble, average='weighted')
    f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
    roc = roc_auc_score(y_test, y_proba_ensemble, multi_class='ovr', average='weighted')
    
    print(f"\n✅ ENSEMBLE RESULTS:")
    print(f"  Accuracy:  {acc:.4f} ({acc:.2%})")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc:.4f}")
    
    # Confusion Matrix
    print("\n" + "-" * 70)
    print("CONFUSION MATRIX (Ensemble)")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_pred_ensemble)
    print(f"\n{'':>10}", end="")
    for cls in classes:
        print(f"{cls:>8}", end="")
    print("  (Predicted)")
    
    for i, cls in enumerate(classes):
        print(f"{cls:>10}", end="")
        for j in range(len(classes)):
            print(f"{cm[i,j]:>8}", end="")
        print()
    print("(Actual)")
    
    # Classification Report
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT (Ensemble)")
    print("-" * 70)
    print(classification_report(y_test, y_pred_ensemble, target_names=classes, digits=4))
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("📋 SUMMARY - ALL MODELS COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 70)
    
    for name, metrics in model_results.items():
        print(f"{name:<15} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['roc_auc']:>10.4f}")
    
    print(f"{'ENSEMBLE':<15} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {roc:>10.4f}")
    print("-" * 70)
    
    # Best model
    best_base = max(model_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Base Model: {best_base[0].upper()} ({best_base[1]['accuracy']:.2%})")
    print(f"🏆 Ensemble:        {acc:.2%}")
    
    if acc >= best_base[1]['accuracy']:
        print(f"\n✅ Ensemble matches or beats the best base model!")
    else:
        diff = best_base[1]['accuracy'] - acc
        print(f"\n⚠️ Best base model is {diff:.2%} better than ensemble")


def test_single_patient(ml_service, explanation_service):
    """Test prediction on a single sample patient."""
    print("\n" + "=" * 70)
    print("🧑‍⚕️ SINGLE PATIENT PREDICTION TEST")
    print("=" * 70)
    
    # Create a sample patient (high-risk example)
    print("\nCreating sample patient...")
    patient = PatientInput(
        age=21464,  # ~58 years
        sex="F",
        drug="D-penicillamine",
        ascites="Y",
        hepatomegaly="Y",
        spiders="Y",
        edema="Y",
        bilirubin=14.5,
        cholesterol=261,
        albumin=2.6,
        copper=156,
        alk_phos=1718,
        sgot=137.95,
        tryglicerides=172,
        platelets=190,
        prothrombin=12.2,
        stage=4
    )
    
    print(f"\nPatient Profile:")
    print(f"  - Age: {patient.age / 365.25:.1f} years")
    print(f"  - Sex: {patient.sex.value}")
    print(f"  - Bilirubin: {patient.bilirubin} mg/dL")
    print(f"  - Albumin: {patient.albumin} g/dL")
    print(f"  - Ascites: {patient.ascites.value}")
    print(f"  - Stage: {patient.stage}")
    
    # Make prediction
    print("\n" + "-" * 70)
    print("ENSEMBLE PREDICTION")
    print("-" * 70)
    
    prediction = ml_service.predict(patient)
    
    print(f"\nFinal Prediction: {prediction['final_prediction']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f"Risk Level: {prediction['risk_level']}")
    print(f"\nProbabilities:")
    for cls, prob in prediction['probabilities'].items():
        print(f"  - {cls}: {prob:.2%}")
    
    print(f"\nBase Model Predictions:")
    for bp in prediction['base_model_predictions']:
        print(f"  - {bp.model_name}: {bp.prediction} ({bp.probability:.2%})")
    
    # Get explanation
    print("\n" + "-" * 70)
    print("INTERPRETABLE EXPLANATION")
    print("-" * 70)
    
    explanation = explanation_service.generate_explanation(patient, prediction)
    
    print(f"\nDecision Tree Rule Path:")
    for i, rule in enumerate(explanation.rule_path, 1):
        print(f"  {i}. {rule}")
    
    print(f"\nTree-Ensemble Agreement: {explanation.tree_ensemble_agreement}")
    
    print(f"\nKey Features Identified:")
    for feature, info in explanation.key_features.items():
        print(f"  - {feature}: {info['formatted']}")


def main():
    """Main entry point for running the full test suite."""
    print("=" * 70)
    print("CIRRHOSIS AI DIAGNOSIS - MODEL EVALUATION & TEST SUITE")
    print("=" * 70)
    
    # Load trained models from backend/models
    print("\n📦 Loading trained models...")
    model_dir = project_root / "backend" / "models"
    ml_service = MLService(model_dir=str(model_dir))
    ml_service.load_models()
    print("✓ Models loaded successfully!")
    
    explanation_service = ExplanationService(ml_service)
    
    # 1. Evaluate on held-out test set
    evaluate_on_test_set(ml_service)
    
    # 2. Test single patient prediction
    test_single_patient(ml_service, explanation_service)
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
