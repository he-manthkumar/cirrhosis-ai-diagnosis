"""
Prediction Router - API endpoints for making predictions and getting explanations.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from backend.models.patient import (
    PatientInput, 
    PredictionResponse, 
    ExplanationResponse
)
from backend.services.ml_service import MLService
from backend.services.explanation_service import ExplanationService
from backend.services.llm_service import LLMService


router = APIRouter(prefix="/predict", tags=["Prediction"])

# Service instances (will be properly initialized with dependency injection)
ml_service: Optional[MLService] = None
explanation_service: Optional[ExplanationService] = None
llm_service: Optional[LLMService] = None


def get_ml_service() -> MLService:
    """Dependency to get ML service."""
    global ml_service
    if ml_service is None:
        ml_service = MLService()
        try:
            ml_service.load_models()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Models not trained yet. Please train the model first via /train endpoint."
            )
    return ml_service


def get_explanation_service(ml_svc: MLService = Depends(get_ml_service)) -> ExplanationService:
    """Dependency to get explanation service."""
    global explanation_service
    if explanation_service is None:
        explanation_service = ExplanationService(ml_svc)
    return explanation_service


def get_llm_service() -> LLMService:
    """Dependency to get LLM service."""
    global llm_service
    if llm_service is None:
        llm_service = LLMService(provider="openai")
    return llm_service


@router.post("/", response_model=PredictionResponse)
async def predict_patient_status(
    patient: PatientInput,
    ml_svc: MLService = Depends(get_ml_service)
) -> PredictionResponse:
    """
    Predict patient status using the stacked ensemble model.
    
    Returns:
        - Final ensemble prediction (C/CL/D)
        - Confidence score
        - Probabilities for each class
        - Individual base model predictions
        - Risk level assessment
    """
    try:
        result = ml_svc.predict(patient)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(
    patient: PatientInput,
    include_narrative: bool = True,
    ml_svc: MLService = Depends(get_ml_service),
    exp_svc: ExplanationService = Depends(get_explanation_service),
    llm_svc: LLMService = Depends(get_llm_service)
) -> ExplanationResponse:
    """
    Get an interpretable explanation for a prediction.
    
    This endpoint:
    1. Makes a prediction using the ensemble
    2. Extracts decision rules from the interpretable Decision Tree
    3. Optionally generates a clinical narrative using an LLM
    
    Returns:
        - Decision rules from the tree
        - Human-readable rule path
        - Key contributing features
        - LLM-generated narrative (if requested)
        - Agreement status between tree and ensemble
    """
    try:
        # Get ensemble prediction
        prediction = ml_svc.predict(patient)
        
        # Generate explanation
        explanation = exp_svc.generate_explanation(patient, prediction)
        
        # Generate LLM narrative if requested
        if include_narrative and llm_svc.is_configured():
            prompt = exp_svc.generate_llm_prompt(patient, prediction, explanation)
            narrative = await llm_svc.generate_narrative(prompt)
            explanation.narrative = narrative
        
        return explanation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full")
async def predict_with_explanation(
    patient: PatientInput,
    ml_svc: MLService = Depends(get_ml_service),
    exp_svc: ExplanationService = Depends(get_explanation_service),
    llm_svc: LLMService = Depends(get_llm_service)
):
    """
    Get both prediction and explanation in a single request.
    
    This is the main endpoint for the complete "Predict and Explain" workflow.
    """
    try:
        # Get prediction
        prediction = ml_svc.predict(patient)
        
        # Get explanation
        explanation = exp_svc.generate_explanation(patient, prediction)
        
        # Generate narrative
        prompt = exp_svc.generate_llm_prompt(patient, prediction, explanation)
        if llm_svc.is_configured():
            narrative = await llm_svc.generate_narrative(prompt)
            explanation.narrative = narrative
        
        return {
            "prediction": PredictionResponse(**prediction),
            "explanation": explanation,
            "llm_prompt": prompt  # Include for debugging/transparency
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance")
async def get_feature_importance(
    exp_svc: ExplanationService = Depends(get_explanation_service)
):
    """
    Get feature importance from the interpretable Decision Tree.
    """
    try:
        importance = exp_svc.get_feature_importance_summary()
        return {"feature_importance": importance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
