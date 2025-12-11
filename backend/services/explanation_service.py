"""
Explanation Service - Generates interpretable explanations for predictions.

This service:
1. Extracts decision rules from the shallow Decision Tree
2. Identifies key contributing features with clinical context
3. Computes SHAP-based feature importance
4. Prepares prompts for LLM narrative generation
5. Provides comprehensive explanation data for frontend display
"""
from typing import Dict, List, Optional, Any
import numpy as np
from backend.models.patient import PatientInput, DecisionRule, ExplanationResponse
from backend.services.ml_service import MLService


class ExplanationService:
    """
    Service for generating interpretable explanations from the ensemble model.
    
    Provides:
    - Decision tree rule paths (deterministic, auditable)
    - Clinical context for feature values
    - Feature importance rankings
    - LLM prompts for narrative generation
    - Risk factor summaries
    """
    
    def __init__(self, ml_service: MLService):
        self.ml_service = ml_service
        
        # Clinical reference ranges for liver disease
        self.reference_ranges = {
            'Bilirubin': {
                'normal': (0.1, 1.2), 
                'unit': 'mg/dL', 
                'high_concern': 3.0,
                'description': 'Liver waste product - elevated indicates liver dysfunction'
            },
            'Albumin': {
                'normal': (3.5, 5.0), 
                'unit': 'g/dL', 
                'low_concern': 2.8,
                'description': 'Protein produced by liver - low indicates poor liver function'
            },
            'Prothrombin': {
                'normal': (11, 13.5), 
                'unit': 'seconds', 
                'high_concern': 15,
                'description': 'Blood clotting time - prolonged indicates coagulation issues'
            },
            'Copper': {
                'normal': (15, 60), 
                'unit': 'µg/day', 
                'high_concern': 140,
                'description': 'Urinary copper - elevated in Wilson\'s disease and cirrhosis'
            },
            'SGOT': {
                'normal': (5, 40), 
                'unit': 'U/L', 
                'high_concern': 100,
                'description': 'Liver enzyme (AST) - elevated indicates liver cell damage'
            },
            'Platelets': {
                'normal': (150, 400), 
                'unit': 'x10³/µL', 
                'low_concern': 100,
                'description': 'Blood clotting cells - low in cirrhosis due to spleen enlargement'
            },
            'Cholesterol': {
                'normal': (125, 200), 
                'unit': 'mg/dL',
                'description': 'Blood lipid - abnormal in liver disease'
            },
            'Alk_Phos': {
                'normal': (44, 147), 
                'unit': 'U/L', 
                'high_concern': 500,
                'description': 'Enzyme from bile ducts - elevated in cholestatic liver disease'
            },
            'Tryglicerides': {
                'normal': (50, 150),
                'unit': 'mg/dL',
                'description': 'Blood fat - affected by liver metabolism'
            },
            'Age': {
                'unit': 'years',
                'description': 'Patient age - older patients have higher mortality risk'
            }
        }
        
        # Risk factor weights for summary
        self.high_risk_indicators = {
            'ascites': 'Y',
            'hepatomegaly': 'Y', 
            'spiders': 'Y',
            'edema': 'S',  # Severe edema
            'stage': 4
        }
    
    def _assess_feature_status(self, feature: str, value: float) -> str:
        """Assess if a feature value is normal, high, or low with severity."""
        if feature not in self.reference_ranges:
            return "unknown"
        
        ref = self.reference_ranges[feature]
        if 'normal' not in ref:
            return "noted"
        
        low, high = ref['normal']
        
        if value < low:
            low_concern = ref.get('low_concern', low * 0.7)
            if value < low_concern:
                return "critically low"
            return "low"
        elif value > high:
            high_concern = ref.get('high_concern', high * 1.5)
            if value > high_concern:
                return "critically elevated"
            return "elevated"
        else:
            return "normal"
    
    def _format_feature_value(self, feature: str, value: float) -> str:
        """Format feature value with clinical context."""
        if feature == 'Age' or feature == 'Age_Years':
            if value > 100:  # Value is in days
                years = value / 365.25
            else:
                years = value
            return f"{years:.1f} years"
        
        ref = self.reference_ranges.get(feature, {})
        unit = ref.get('unit', '')
        status = self._assess_feature_status(feature, value)
        
        return f"{value:.2f} {unit} ({status})"
    
    def _get_risk_factors(self, patient: PatientInput) -> List[Dict[str, Any]]:
        """Identify risk factors present in the patient."""
        risk_factors = []
        
        # Check clinical signs
        if patient.ascites and patient.ascites.value == 'Y':
            risk_factors.append({
                'factor': 'Ascites',
                'value': 'Present',
                'severity': 'high',
                'explanation': 'Fluid accumulation in abdomen indicates advanced liver disease'
            })
        
        if patient.hepatomegaly and patient.hepatomegaly.value == 'Y':
            risk_factors.append({
                'factor': 'Hepatomegaly',
                'value': 'Present', 
                'severity': 'moderate',
                'explanation': 'Enlarged liver suggests ongoing liver damage'
            })
        
        if patient.spiders and patient.spiders.value == 'Y':
            risk_factors.append({
                'factor': 'Spider Angiomas',
                'value': 'Present',
                'severity': 'moderate',
                'explanation': 'Spider-like blood vessels indicate portal hypertension'
            })
        
        if patient.edema:
            if patient.edema.value in ['S', 'Y']:
                risk_factors.append({
                    'factor': 'Edema',
                    'value': 'Present' if patient.edema.value == 'Y' else 'Severe',
                    'severity': 'high' if patient.edema.value == 'S' else 'moderate',
                    'explanation': 'Fluid retention indicates liver and/or kidney dysfunction'
                })
        
        if patient.stage and patient.stage >= 4:
            risk_factors.append({
                'factor': 'Disease Stage',
                'value': f'Stage {patient.stage}',
                'severity': 'high',
                'explanation': 'Advanced histological stage indicates severe liver damage'
            })
        
        # Check lab values
        if patient.bilirubin > 3.0:
            risk_factors.append({
                'factor': 'Elevated Bilirubin',
                'value': f'{patient.bilirubin:.1f} mg/dL',
                'severity': 'high' if patient.bilirubin > 5.0 else 'moderate',
                'explanation': 'High bilirubin indicates impaired liver excretion'
            })
        
        if patient.albumin < 2.8:
            risk_factors.append({
                'factor': 'Low Albumin',
                'value': f'{patient.albumin:.1f} g/dL',
                'severity': 'high' if patient.albumin < 2.5 else 'moderate',
                'explanation': 'Low albumin indicates reduced liver synthetic function'
            })
        
        if patient.prothrombin > 15:
            risk_factors.append({
                'factor': 'Prolonged Prothrombin',
                'value': f'{patient.prothrombin:.1f} seconds',
                'severity': 'high' if patient.prothrombin > 17 else 'moderate',
                'explanation': 'Prolonged clotting time indicates impaired coagulation factor production'
            })
        
        if patient.platelets and patient.platelets < 100:
            risk_factors.append({
                'factor': 'Low Platelets',
                'value': f'{patient.platelets:.0f} x10³/µL',
                'severity': 'moderate',
                'explanation': 'Thrombocytopenia common in cirrhosis due to spleen sequestration'
            })
        
        return risk_factors
    
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
                'status': self._assess_feature_status(feature, actual),
                'description': self.reference_ranges.get(feature, {}).get('description', '')
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
    
    def generate_comprehensive_explanation(
        self,
        patient: PatientInput,
        ensemble_prediction: Dict
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation including all components.
        
        Returns a dictionary with:
        - Basic explanation (rules, key features)
        - Risk factors summary
        - Feature importance
        - Clinical interpretation
        - LLM prompt
        """
        # Get basic explanation
        explanation = self.generate_explanation(patient, ensemble_prediction)
        
        # Get risk factors
        risk_factors = self._get_risk_factors(patient)
        
        # Get feature importance
        feature_importance = self.get_feature_importance_summary()
        
        # Count risk severity
        high_risk_count = sum(1 for rf in risk_factors if rf['severity'] == 'high')
        moderate_risk_count = sum(1 for rf in risk_factors if rf['severity'] == 'moderate')
        
        # Generate risk summary
        if high_risk_count >= 3:
            risk_summary = "Multiple high-risk factors present - elevated mortality risk"
        elif high_risk_count >= 1:
            risk_summary = "Some high-risk factors present - increased concern"
        elif moderate_risk_count >= 2:
            risk_summary = "Moderate risk factors present - monitor closely"
        else:
            risk_summary = "Limited risk factors identified - relatively favorable prognosis"
        
        # Build comprehensive response
        return {
            'prediction': {
                'status': ensemble_prediction['final_prediction'],
                'confidence': ensemble_prediction['confidence'],
                'risk_level': ensemble_prediction['risk_level'],
                'probabilities': ensemble_prediction['probabilities']
            },
            'decision_tree': {
                'rule_path': explanation.rule_path,
                'decision_rules': [
                    {
                        'feature': rule.feature,
                        'operator': rule.operator,
                        'threshold': rule.threshold,
                        'direction': rule.direction
                    }
                    for rule in explanation.decision_rules
                ],
                'tree_ensemble_agreement': explanation.tree_ensemble_agreement
            },
            'key_features': explanation.key_features,
            'risk_factors': risk_factors,
            'risk_summary': risk_summary,
            'feature_importance': feature_importance,
            'base_model_predictions': [
                {
                    'model': bp.model_name,
                    'prediction': bp.prediction,
                    'probability': bp.probability
                }
                for bp in ensemble_prediction['base_model_predictions']
            ],
            'model_agreement': all(
                bp.prediction == ensemble_prediction['final_prediction']
                for bp in ensemble_prediction['base_model_predictions']
            )
        }
    
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
- Ascites: {patient.ascites.value if patient.ascites else 'N'}
- Hepatomegaly: {patient.hepatomegaly.value if patient.hepatomegaly else 'N'}
- Edema: {patient.edema.value if patient.edema else 'N'}
"""
        
        # Get risk factors
        risk_factors = self._get_risk_factors(patient)
        risk_text = ""
        if risk_factors:
            risk_items = [f"- {rf['factor']}: {rf['value']} ({rf['explanation']})" for rf in risk_factors]
            risk_text = f"\nIdentified Risk Factors:\n" + "\n".join(risk_items)
        
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
