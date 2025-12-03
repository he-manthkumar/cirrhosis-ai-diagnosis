"""
Script to train the Stacked Ensemble model.

Run this script to train all base models and the meta-model:
    python -m scripts.train_model

Note: Run preprocess_data.py first if you haven't already.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.ml_service import MLService


def main():
    print("=" * 60)
    print("CIRRHOSIS AI DIAGNOSIS - MODEL TRAINING")
    print("=" * 60)
    
    # Check if preprocessed data exists (try multiple possible paths)
    possible_paths = [
        project_root / "data" / "processed" / "cirrhosis_cleaned.csv",
        Path(r"D:\Finalyearproject\cirrhosis-ai-diagnosis\data\processed\cirrhosis_cleaned.csv"),
        Path(r"cirrhosis-ai-diagnosis\data\processed\cirrhosis_cleaned.csv"),
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = str(path)
            print(f"\n✓ Using preprocessed data: {data_path}")
            break
    
    if data_path is None:
        print("\n❌ Error: Preprocessed data not found!")
        print("   Please run the preprocessing notebook first.")
        print("   Expected location: data/processed/cirrhosis_cleaned.csv")
        sys.exit(1)
    
    print("\nTraining Interpretable Stacked Ensemble...")
    print("\nArchitecture:")
    print("  Layer 1 (Base Models):")
    print("    - XGBoost Classifier")
    print("    - LightGBM Classifier")
    print("    - CatBoost Classifier")
    print("    - Decision Tree Classifier (Interpreter, max_depth=4)")
    print("  Layer 2 (Meta-Model):")
    print("    - Logistic Regression")
    print("\n" + "-" * 60)
    
    # Initialize and train - models saved to backend/models/
    model_dir = project_root / "backend" / "models"
    ml_service = MLService(model_dir=str(model_dir))
    result = ml_service.train(data_path=data_path, test_size=0.2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    
    print(f"\n📊 Dataset Split:")
    print(f"  - Training samples: {result['train_samples']}")
    print(f"  - Test samples: {result['test_samples']} (held-out, never seen during training)")
    print(f"  - Number of Features: {result['n_features']}")
    print(f"  - Classes: {result['classes']}")
    
    print(f"\n🎯 HONEST Test Set Performance (on unseen data):")
    print(f"  ┌{'─'*40}┐")
    print(f"  │ Ensemble Accuracy:  {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)      │")
    print(f"  │ Ensemble Precision: {result['test_precision']:.4f}                │")
    print(f"  │ Ensemble Recall:    {result['test_recall']:.4f}                │")
    print(f"  │ Ensemble F1-Score:  {result['test_f1']:.4f}                │")
    print(f"  │ Ensemble ROC-AUC:   {result['test_roc_auc']:.4f}                │")
    print(f"  └{'─'*40}┘")
    
    print(f"\n📈 Base Model Test Accuracy:")
    for name, metrics in result['test_base_metrics'].items():
        print(f"  - {name}: {metrics['accuracy']:.4f} (F1: {metrics['f1']:.4f})")
    
    print(f"\n✓ Models saved to: backend/models/")
    print(f"✓ Test data saved for notebook evaluation")
    print("\n⚠️  These are REAL metrics on data the model never saw during training!")
    
if __name__ == "__main__":
    main()
