"""
Training Router - API endpoints for model training and management.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
from pydantic import BaseModel

from backend.services.ml_service import MLService


router = APIRouter(prefix="/train", tags=["Training"])

# Track training status
training_status = {
    "is_training": False,
    "last_result": None,
    "error": None
}


class TrainingRequest(BaseModel):
    """Request schema for training."""
    data_path: str = "data/processed/cirrhosis_cleaned.csv"


class TrainingResponse(BaseModel):
    """Response schema for training."""
    status: str
    message: str
    details: Optional[dict] = None


@router.post("/", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> TrainingResponse:
    """
    Train the stacked ensemble model.
    
    This trains:
    - XGBoost, LightGBM, CatBoost (performance models)
    - Decision Tree (interpreter model)
    - Logistic Regression (meta-model)
    
    Training uses 5-fold cross-validation for generating meta-features.
    """
    global training_status
    
    if training_status["is_training"]:
        return TrainingResponse(
            status="in_progress",
            message="Training is already in progress. Please wait."
        )
    
    # Start training in background
    background_tasks.add_task(run_training, request.data_path)
    training_status["is_training"] = True
    training_status["error"] = None
    
    return TrainingResponse(
        status="started",
        message="Training started. Use /train/status to check progress."
    )


def run_training(data_path: str):
    """Background task to run training."""
    global training_status
    
    try:
        ml_service = MLService()
        result = ml_service.train(data_path)
        
        training_status["last_result"] = result
        training_status["is_training"] = False
        training_status["error"] = None
        
    except Exception as e:
        training_status["is_training"] = False
        training_status["error"] = str(e)
        training_status["last_result"] = None


@router.get("/status")
async def get_training_status():
    """
    Get the current training status.
    """
    return {
        "is_training": training_status["is_training"],
        "last_result": training_status["last_result"],
        "error": training_status["error"]
    }


@router.post("/sync", response_model=TrainingResponse)
async def train_model_sync(request: TrainingRequest) -> TrainingResponse:
    """
    Train the model synchronously (blocking).
    
    Use this for testing or when you need to wait for training to complete.
    """
    try:
        ml_service = MLService()
        result = ml_service.train(request.data_path)
        
        return TrainingResponse(
            status="completed",
            message="Training completed successfully!",
            details=result
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {request.data_path}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@router.get("/metrics")
async def get_model_metrics():
    """
    Get metrics from the last training run.
    """
    if training_status["last_result"] is None:
        raise HTTPException(
            status_code=404,
            detail="No training results available. Train the model first."
        )
    
    return training_status["last_result"]
