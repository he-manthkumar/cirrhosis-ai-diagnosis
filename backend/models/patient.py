"""
Pydantic models for patient data and prediction responses.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class StatusEnum(str, Enum):
    """Patient status outcomes."""
    C = "C"      # Censored (alive at end of study)
    CL = "CL"    # Censored due to liver transplant
    D = "D"      # Death


class DrugEnum(str, Enum):
    """Drug treatment options."""
    D_PENICILLAMINE = "D-penicillamine"
    PLACEBO = "Placebo"


class SexEnum(str, Enum):
    """Patient sex."""
    M = "M"
    F = "F"


class BinaryEnum(str, Enum):
    """Yes/No binary options."""
    Y = "Y"
    N = "N"


class EdemaEnum(str, Enum):
    """Edema status."""
    N = "N"      # No edema
    S = "S"      # Edema without diuretics
    Y = "Y"      # Edema despite diuretic therapy


class PatientInput(BaseModel):
    """Input schema for patient clinical data."""
    
    # Demographics
    age: float = Field(..., description="Age in days", ge=0)
    sex: SexEnum = Field(..., description="Patient sex (M/F)")
    
    # Treatment
    drug: Optional[DrugEnum] = Field(None, description="Drug treatment")
    
    # Clinical Signs
    ascites: BinaryEnum = Field(..., description="Presence of ascites (Y/N)")
    hepatomegaly: BinaryEnum = Field(..., description="Presence of hepatomegaly (Y/N)")
    spiders: BinaryEnum = Field(..., description="Presence of spider angiomas (Y/N)")
    edema: EdemaEnum = Field(..., description="Edema status (N/S/Y)")
    
    # Lab Values
    bilirubin: float = Field(..., description="Serum bilirubin (mg/dl)", ge=0)
    cholesterol: Optional[float] = Field(None, description="Serum cholesterol (mg/dl)", ge=0)
    albumin: float = Field(..., description="Albumin (gm/dl)", ge=0)
    copper: Optional[float] = Field(None, description="Urine copper (ug/day)", ge=0)
    alk_phos: Optional[float] = Field(None, description="Alkaline phosphatase (U/liter)", ge=0)
    sgot: Optional[float] = Field(None, description="SGOT (U/ml)", ge=0)
    tryglicerides: Optional[float] = Field(None, description="Triglycerides (mg/dl)", ge=0)
    platelets: Optional[float] = Field(None, description="Platelets per cubic ml/1000", ge=0)
    prothrombin: float = Field(..., description="Prothrombin time (seconds)", ge=0)
    
    # Disease Stage
    stage: Optional[int] = Field(None, description="Histologic stage (1-4)", ge=1, le=4)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 21464,
                "sex": "F",
                "drug": "D-penicillamine",
                "ascites": "Y",
                "hepatomegaly": "Y",
                "spiders": "Y",
                "edema": "Y",
                "bilirubin": 14.5,
                "cholesterol": 261,
                "albumin": 2.6,
                "copper": 156,
                "alk_phos": 1718,
                "sgot": 137.95,
                "tryglicerides": 172,
                "platelets": 190,
                "prothrombin": 12.2,
                "stage": 4
            }
        }


class DecisionRule(BaseModel):
    """A single decision rule from the interpretable tree."""
    feature: str
    operator: str  # "<=", ">", "=="
    threshold: float
    direction: str  # "left" or "right"


class BaseModelPrediction(BaseModel):
    """Prediction from a single base model."""
    model_name: str
    prediction: str
    probability: float
    confidence: float


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    
    # Final ensemble prediction
    final_prediction: str = Field(..., description="Final predicted status (C/CL/D)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    
    # Probabilities for each class
    probabilities: dict = Field(..., description="Probability for each class")
    
    # Individual base model predictions
    base_model_predictions: List[BaseModelPrediction] = Field(
        ..., description="Predictions from each base model"
    )
    
    # Risk assessment
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")


class ExplanationResponse(BaseModel):
    """Response schema for model explanations."""
    
    # Decision tree rules
    decision_rules: List[DecisionRule] = Field(
        ..., description="Decision rules from interpretable tree"
    )
    
    # Rule path as human-readable text
    rule_path: List[str] = Field(
        ..., description="Human-readable decision path"
    )
    
    # Key contributing features
    key_features: dict = Field(
        ..., description="Most important features for this prediction"
    )
    
    # LLM-generated narrative (optional, filled by LLM service)
    narrative: Optional[str] = Field(
        None, description="LLM-generated clinical narrative"
    )
    
    # Agreement between tree and ensemble
    tree_ensemble_agreement: bool = Field(
        ..., description="Whether tree prediction matches ensemble"
    )
