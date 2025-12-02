"""
Helper utilities for the Cirrhosis AI Diagnosis system.
"""
from typing import Dict, Optional, Tuple


def convert_age_to_years(age_days: float) -> float:
    """Convert age from days to years."""
    return age_days / 365.25


def convert_age_to_days(age_years: float) -> float:
    """Convert age from years to days."""
    return age_years * 365.25


# Clinical reference ranges for cirrhosis-related lab values
CLINICAL_REFERENCE_RANGES: Dict[str, Dict] = {
    'Bilirubin': {
        'normal_range': (0.1, 1.2),
        'unit': 'mg/dL',
        'clinical_significance': 'Elevated bilirubin indicates impaired liver function or bile duct obstruction'
    },
    'Albumin': {
        'normal_range': (3.5, 5.0),
        'unit': 'g/dL',
        'clinical_significance': 'Low albumin indicates impaired liver synthetic function'
    },
    'Prothrombin': {
        'normal_range': (11.0, 13.5),
        'unit': 'seconds',
        'clinical_significance': 'Prolonged PT indicates impaired coagulation factor synthesis'
    },
    'Copper': {
        'normal_range': (15, 60),
        'unit': 'µg/day (urine)',
        'clinical_significance': 'Elevated copper may indicate Wilson disease or cholestatic liver disease'
    },
    'SGOT': {
        'normal_range': (5, 40),
        'unit': 'U/L',
        'clinical_significance': 'Elevated SGOT (AST) indicates hepatocellular damage'
    },
    'Platelets': {
        'normal_range': (150, 400),
        'unit': 'x10³/µL',
        'clinical_significance': 'Low platelets may indicate portal hypertension or hypersplenism'
    },
    'Cholesterol': {
        'normal_range': (125, 200),
        'unit': 'mg/dL',
        'clinical_significance': 'Abnormal cholesterol may indicate metabolic dysfunction'
    },
    'Alk_Phos': {
        'normal_range': (44, 147),
        'unit': 'U/L',
        'clinical_significance': 'Elevated ALP indicates cholestasis or bone disease'
    },
    'Tryglicerides': {
        'normal_range': (50, 150),
        'unit': 'mg/dL',
        'clinical_significance': 'Elevated triglycerides indicate metabolic dysfunction'
    }
}


def validate_clinical_ranges(values: Dict[str, float]) -> Dict[str, Dict]:
    """
    Validate clinical values against reference ranges.
    
    Args:
        values: Dictionary of feature names to values
        
    Returns:
        Dictionary with validation results for each feature
    """
    results = {}
    
    for feature, value in values.items():
        if feature not in CLINICAL_REFERENCE_RANGES:
            results[feature] = {
                'value': value,
                'status': 'unknown',
                'message': 'No reference range available'
            }
            continue
        
        ref = CLINICAL_REFERENCE_RANGES[feature]
        low, high = ref['normal_range']
        
        if value < low:
            status = 'low'
            message = f'Below normal range ({low}-{high} {ref["unit"]})'
        elif value > high:
            status = 'high'
            message = f'Above normal range ({low}-{high} {ref["unit"]})'
        else:
            status = 'normal'
            message = f'Within normal range ({low}-{high} {ref["unit"]})'
        
        results[feature] = {
            'value': value,
            'status': status,
            'unit': ref['unit'],
            'normal_range': ref['normal_range'],
            'message': message,
            'clinical_significance': ref['clinical_significance']
        }
    
    return results


def get_risk_factors(patient_data: Dict) -> Dict[str, str]:
    """
    Identify risk factors from patient data.
    
    Returns a dictionary of identified risk factors and their severity.
    """
    risk_factors = {}
    
    # Check bilirubin
    if patient_data.get('bilirubin', 0) > 3.0:
        risk_factors['high_bilirubin'] = 'severe'
    elif patient_data.get('bilirubin', 0) > 1.2:
        risk_factors['elevated_bilirubin'] = 'moderate'
    
    # Check albumin
    if patient_data.get('albumin', 5) < 2.8:
        risk_factors['low_albumin'] = 'severe'
    elif patient_data.get('albumin', 5) < 3.5:
        risk_factors['reduced_albumin'] = 'moderate'
    
    # Check prothrombin time
    if patient_data.get('prothrombin', 11) > 15:
        risk_factors['prolonged_pt'] = 'severe'
    elif patient_data.get('prothrombin', 11) > 13.5:
        risk_factors['elevated_pt'] = 'moderate'
    
    # Check for ascites
    if patient_data.get('ascites') == 'Y':
        risk_factors['ascites_present'] = 'significant'
    
    # Check for hepatomegaly
    if patient_data.get('hepatomegaly') == 'Y':
        risk_factors['hepatomegaly'] = 'moderate'
    
    # Check for edema
    if patient_data.get('edema') == 'Y':
        risk_factors['edema_refractory'] = 'severe'
    elif patient_data.get('edema') == 'S':
        risk_factors['edema_present'] = 'moderate'
    
    # Check platelets
    if patient_data.get('platelets', 200) < 100:
        risk_factors['thrombocytopenia'] = 'severe'
    elif patient_data.get('platelets', 200) < 150:
        risk_factors['low_platelets'] = 'moderate'
    
    return risk_factors


def calculate_meld_score(
    bilirubin: float,
    creatinine: float,
    inr: float,
    sodium: Optional[float] = None
) -> Tuple[float, str]:
    """
    Calculate MELD score (Model for End-Stage Liver Disease).
    
    Note: This is a simplified version. The actual MELD score requires
    creatinine and INR which are not in the PBC dataset.
    
    Args:
        bilirubin: Serum bilirubin (mg/dL)
        creatinine: Serum creatinine (mg/dL)
        inr: International Normalized Ratio
        sodium: Serum sodium (mEq/L) - for MELD-Na
        
    Returns:
        Tuple of (MELD score, severity category)
    """
    import math
    
    # Minimum values per MELD calculation rules
    bilirubin = max(1.0, bilirubin)
    creatinine = max(1.0, min(4.0, creatinine))  # Cap at 4.0
    inr = max(1.0, inr)
    
    # MELD Score formula
    meld = (
        0.957 * math.log(creatinine) +
        0.378 * math.log(bilirubin) +
        1.120 * math.log(inr) +
        0.643
    ) * 10
    
    meld = round(meld)
    
    # Categorize severity
    if meld < 10:
        severity = "Low (good prognosis)"
    elif meld < 20:
        severity = "Moderate"
    elif meld < 30:
        severity = "High"
    else:
        severity = "Very High (urgent)"
    
    return meld, severity
