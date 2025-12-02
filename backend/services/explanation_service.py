"""
Explanation Service - Generates interpretable explanations for predictions.

This service:
1. Extracts decision rules from the shallow Decision Tree
2. Identifies key contributing features
3. Prepares prompts for LLM narrative generation
"""
from typing import Dict, List, Optional
from backend.models.patient import PatientInput, DecisionRule, ExplanationResponse
from backend.services.ml_service import MLService


class ExplanationService:
    """
    Service for generating interpretable explanations from the ensemble model.
    """
    
    def __init__(self, ml_service: MLService):
        self.ml_service = ml_service
        
        # Clinical reference ranges for context
        self.reference_ranges = {
            'Bilirubin': {'normal': (0.1, 1.2), 'unit': 'mg/dL', 'high_concern': 3.0},
            'Albumin': {'normal': (3.5, 5.0), 'unit': 'g/dL', 'low_concern': 2.8},
            'Prothrombin': {'normal': (11, 13.5), 'unit': 'seconds', 'high_concern': 15},
            'Copper': {'normal': (15, 60), 'unit': 'µg/day', 'high_concern': 140},
            'SGOT': {'normal': (5, 40), 'unit': 'U/L', 'high_concern': 100},
            'Platelets': {'normal': (150, 400), 'unit': 'x10³/µL', 'low_concern': 100},
            'Cholesterol': {'normal': (125, 200), 'unit': 'mg/dL'},
            'Alk_Phos': {'normal': (44, 147), 'unit': 'U/L', 'high_concern': 500},
            'Age': {'unit': 'days'}  # Will convert to years
        }
    
    def _assess_feature_status(self, feature: str, value: float) -> str:
        """Assess if a feature value is normal, high, or low."""
        if feature not in self.reference_ranges:
            return "unknown"
        
        ref = self.reference_ranges[feature]
        if 'normal' not in ref:
            return "noted"
        
        low, high = ref['normal']
        if value < low:
            return "low"
        elif value > high:
            return "elevated"
        else:
            return "normal"
    
    def _format_feature_value(self, feature: str, value: float) -> str:
        """Format feature value with clinical context."""
        if feature == 'Age':
            years = value / 365.25
            return f"{years:.1f} years"
        
        ref = self.reference_ranges.get(feature, {})
        unit = ref.get('unit', '')
        status = self._assess_feature_status(feature, value)
        
        return f"{value:.2f} {unit} ({status})"
    
    def generate_explanation(
        self, 
        patient: PatientInput,
        ensemble_prediction: Dict
    ) -> ExplanationResponse:
        """
        Generate a complete explanation for a prediction.
        
        Args:
            patient: Patient input data
            ensemble_prediction: Prediction from the ensemble model
            
        Returns:
            ExplanationResponse with rules, key features, and narrative prompt
        """
        # Get decision tree path
        tree_result = self.ml_service.get_decision_tree_path(patient)
        
        # Convert to DecisionRule objects
        decision_rules = [
            DecisionRule(
                feature=rule['feature'],
                operator=rule['operator'],
                threshold=rule['threshold'],
                direction=rule['direction']
            )
            for rule in tree_result['decision_rules']
        ]
        
        # Identify key features from the decision path
        key_features = {}
        for rule in tree_result['decision_rules']:
            feature = rule['feature']
            actual = rule['actual_value']
            key_features[feature] = {
                'value': actual,
                'formatted': self._format_feature_value(feature, actual),
                'threshold_used': rule['threshold'],
                'status': self._assess_feature_status(feature, actual)
            }
        
        # Check tree-ensemble agreement
        tree_prediction = tree_result['tree_prediction']
        ensemble_prediction_status = ensemble_prediction['final_prediction']
        agreement = tree_prediction == ensemble_prediction_status
        
        return ExplanationResponse(
            decision_rules=decision_rules,
            rule_path=tree_result['rule_path'],
            key_features=key_features,
            narrative=None,  # Will be filled by LLM service
            tree_ensemble_agreement=agreement
        )
    
    def generate_llm_prompt(
        self,
        patient: PatientInput,
        ensemble_prediction: Dict,
        explanation: ExplanationResponse
    ) -> str:
        """
        Generate a prompt for the LLM to create a clinical narrative.
        
        This is the key to the "grounded explanation" approach - we give the LLM
        specific rules extracted from the model, not asking it to interpret freely.
        """
        # Format patient context
        age_years = patient.age / 365.25
        patient_context = f"""
Patient Profile:
- Age: {age_years:.1f} years
- Sex: {patient.sex.value}
- Drug Treatment: {patient.drug.value if patient.drug else 'Not specified'}
- Disease Stage: {patient.stage if patient.stage else 'Not specified'}
"""
        
        # Format clinical values
        clinical_values = f"""
Key Clinical Values:
- Bilirubin: {self._format_feature_value('Bilirubin', patient.bilirubin)}
- Albumin: {self._format_feature_value('Albumin', patient.albumin)}
- Prothrombin: {self._format_feature_value('Prothrombin', patient.prothrombin)}
- Ascites: {patient.ascites.value}
- Hepatomegaly: {patient.hepatomegaly.value}
- Edema: {patient.edema.value}
"""
        
        # Format decision rules
        rules_text = "\n".join([f"- {rule}" for rule in explanation.rule_path])
        
        # Build the prompt
        prompt = f"""You are an expert medical AI assistant designed to make machine learning model predictions understandable to clinicians.

An advanced ensemble model (combining XGBoost, LightGBM, CatBoost, and a Decision Tree) predicted that this patient has status: **{ensemble_prediction['final_prediction']}** with {ensemble_prediction['confidence']*100:.1f}% confidence.

Status meanings:
- C: Censored (patient survived to end of study)
- CL: Censored due to liver transplant
- D: Death

{patient_context}
{clinical_values}

To understand this prediction, we analyzed the interpretable Decision Tree component of the model, which made its classification based on the following clinical logic:
{rules_text}

The Decision Tree's prediction {'agrees' if explanation.tree_ensemble_agreement else 'differs from'} the ensemble's final prediction.

Based ONLY on these extracted rules and the patient's clinical values, please generate a concise, one-paragraph narrative (3-5 sentences) explaining the model's reasoning in clear clinical terms. 

Focus on:
1. Which clinical factors were most decisive
2. How the patient's values compare to clinical thresholds
3. Why these factors led to the predicted outcome

Do not add information beyond what is provided. Ground your explanation in the specific rules and values given."""

        return prompt
    
    def get_feature_importance_summary(self) -> Dict:
        """
        Get a summary of feature importance from the decision tree.
        """
        tree = self.ml_service.decision_tree
        importances = tree.feature_importances_
        feature_names = self.ml_service.feature_names_
        
        # Sort by importance
        importance_dict = {
            name: float(imp) 
            for name, imp in zip(feature_names, importances)
            if imp > 0
        }
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance
