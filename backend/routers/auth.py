"""
Authentication Router - signup/login endpoints with basic JWT.
"""
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.auth_database import get_auth_db
from backend.models.auth_models import User
from backend.services.auth_service import (
    authenticate_user,
    create_access_token,
    get_user_by_username,
    hash_password,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/signup", response_model=TokenResponse)
def signup(request: SignupRequest, db: Session = Depends(get_auth_db)):
    existing_user = get_user_by_username(db, request.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )

    user = User(
        username=request.username,
        password_hash=hash_password(request.password)
    )
    db.add(user)
    db.commit()

    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_auth_db)):
    user = authenticate_user(db, request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token)
