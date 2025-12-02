"""
Script to test prediction and explanation workflow.

Run this script to test the full prediction + explanation pipeline:
    python -m scripts.test_prediction
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.models.patient import PatientInput
from backend.services.ml_service import MLService
from backend.services.explanation_service import ExplanationService


def main():
    print("=" * 60)
    print("CIRRHOSIS AI DIAGNOSIS - PREDICTION TEST")
    print("=" * 60)
    
    # Load trained models
    print("\nLoading trained models...")
    ml_service = MLService(model_dir="data/processed")
    ml_service.load_models()
    
    explanation_service = ExplanationService(ml_service)
    
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
    print("\n" + "-" * 60)
    print("ENSEMBLE PREDICTION")
    print("-" * 60)
    
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
    print("\n" + "-" * 60)
    print("INTERPRETABLE EXPLANATION")
    print("-" * 60)
    
    explanation = explanation_service.generate_explanation(patient, prediction)
    
    print(f"\nDecision Tree Rule Path:")
    for i, rule in enumerate(explanation.rule_path, 1):
        print(f"  {i}. {rule}")
    
    print(f"\nTree-Ensemble Agreement: {explanation.tree_ensemble_agreement}")
    
    print(f"\nKey Features Identified:")
    for feature, info in explanation.key_features.items():
        print(f"  - {feature}: {info['formatted']}")
    
    # Generate LLM prompt
    print("\n" + "-" * 60)
    print("LLM PROMPT (for narrative generation)")
    print("-" * 60)
    
    prompt = explanation_service.generate_llm_prompt(patient, prediction, explanation)
    print(prompt)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
