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
    
    # Check if preprocessed data exists
    processed_path = Path("data/processed/cirrhosis_cleaned.csv")
    raw_path = Path("data/raw/cirrhosis.csv")
    
    if processed_path.exists():
        data_path = str(processed_path)
        print(f"\n✓ Using preprocessed data: {data_path}")
    else:
        data_path = str(raw_path)
        print(f"\n⚠ Preprocessed data not found. Using raw data: {data_path}")
        print("  Tip: Run 'python -m scripts.preprocess_data' first for better results.")
    
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
    
    # Initialize and train
    ml_service = MLService(model_dir="data/processed")
    result = ml_service.train(data_path=data_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  - Training Accuracy: {result['accuracy']:.4f}")
    print(f"  - Number of Samples: {result['n_samples']}")
    print(f"  - Number of Features: {result['n_features']}")
    print(f"  - Classes: {result['classes']}")
    print(f"  - Base Models: {result['base_models']}")
    print(f"\nModels saved to: data/processed/")
    

if __name__ == "__main__":
    main()
