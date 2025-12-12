"""
Prediction Router - API endpoints for making predictions and getting explanations.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
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


class ImageInput(BaseModel):
    """Optional image input for external symptom analysis."""
    image_base64: Optional[str] = Field(
        None, 
        description="Base64-encoded image of external symptoms (e.g., jaundice, spider angiomas)"
    )
    image_mime_type: str = Field(
        "image/jpeg",
        description="MIME type of the image (image/jpeg, image/png, etc.)"
    )


class PredictWithImageRequest(BaseModel):
    """Request model for prediction with optional image."""
    patient: PatientInput
    image: Optional[ImageInput] = None


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
    image_base64: Optional[str] = None,
    image_mime_type: str = "image/jpeg",
    ml_svc: MLService = Depends(get_ml_service),
    exp_svc: ExplanationService = Depends(get_explanation_service),
    llm_svc: LLMService = Depends(get_llm_service)
) -> ExplanationResponse:
    """
    Get an interpretable explanation for a prediction.
    
    This endpoint:
    1. Makes a prediction using the ensemble
    2. Extracts decision rules from the interpretable Decision Tree
    3. Optionally generates a clinical narrative using OpenAI LLM
    4. Optionally analyzes external symptom image (lower weightage)
    
    Args:
        patient: Patient clinical data
        include_narrative: Whether to generate LLM narrative
        image_base64: Optional base64-encoded image of external symptoms
        image_mime_type: MIME type of the image
    
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
            narrative = await llm_svc.generate_narrative(
                prompt, 
                image_base64=image_base64,
                image_mime_type=image_mime_type
            )
            explanation.narrative = narrative
        
        return explanation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full")
async def predict_with_explanation(
    request: PredictWithImageRequest,
    ml_svc: MLService = Depends(get_ml_service),
    exp_svc: ExplanationService = Depends(get_explanation_service),
    llm_svc: LLMService = Depends(get_llm_service)
):
    """
    Get both prediction and explanation in a single request.
    
    This is the main endpoint for the complete "Predict and Explain" workflow.
    Supports optional image input for external symptom analysis.
    
    The image (if provided) will be analyzed by OpenAI and integrated into
    the narrative with lower weightage (~20%) compared to clinical data (~80%).
    """
    try:
        patient = request.patient
        image_base64 = request.image.image_base64 if request.image else None
        image_mime_type = request.image.image_mime_type if request.image else "image/jpeg"
        
        # Get prediction
        prediction = ml_svc.predict(patient)
        
        # Get explanation
        explanation = exp_svc.generate_explanation(patient, prediction)
        
        # Generate narrative with optional image
        prompt = exp_svc.generate_llm_prompt(patient, prediction, explanation)
        image_analysis = None
        
        if llm_svc.is_configured():
            narrative = await llm_svc.generate_narrative(
                prompt,
                image_base64=image_base64,
                image_mime_type=image_mime_type
            )
            explanation.narrative = narrative
            
            # If image was provided, also get standalone image analysis
            if image_base64:
                image_analysis = await llm_svc.analyze_image_only(
                    image_base64,
                    image_mime_type
                )
        
        response = {
            "prediction": PredictionResponse(**prediction),
            "explanation": explanation,
            "llm_prompt": prompt  # Include for debugging/transparency
        }
        
        if image_analysis:
            response["image_analysis"] = image_analysis
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-image")
async def analyze_external_symptoms(
    image_base64: str,
    image_mime_type: str = "image/jpeg",
    llm_svc: LLMService = Depends(get_llm_service)
):
    """
    Analyze an external symptom image independently.
    
    This endpoint can be used to get image-based findings before
    or independently of clinical data predictions.
    
    Note: Image findings should be considered supplementary (~20% weight)
    to clinical lab values and model predictions (~80% weight).
    """
    if not llm_svc.is_configured():
        raise HTTPException(
            status_code=503,
            detail="LLM service not configured. Please set OPENAI_API_KEY in environment."
        )
    
    try:
        result = await llm_svc.analyze_image_only(image_base64, image_mime_type)
        return result
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
