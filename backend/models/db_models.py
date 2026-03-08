
# backend/models/db_models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime,timezone
from backend.database import Base

class PatientRecord(Base):
    __tablename__ = "patient_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String, index=True, nullable=False)  # Indexed for fast searching
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Store the input form data
    clinical_data = Column(JSON, nullable=False)
    
    # Store the AI outputs
    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    narrative = Column(String, nullable=True)