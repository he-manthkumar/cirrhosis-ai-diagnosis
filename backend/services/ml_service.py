"""
Machine Learning Service - Interpretable Stacked Ensemble for Cirrhosis Diagnosis.

Architecture:
    Layer 1 (Base Models):
        - XGBoost Classifier (Performance)
        - LightGBM Classifier (Performance)  
        - CatBoost Classifier (Performance)
        - Decision Tree Classifier (Interpreter) - max_depth=4
    
    Layer 2 (Meta-Model):
        - Logistic Regression
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from backend.models.patient import PatientInput, BaseModelPrediction


class MLService:
    """
    Interpretable Stacked Ensemble for Cirrhosis Patient Status Prediction.
    """
    
    def __init__(self, model_dir: str = "data/processed"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature columns
        self.numerical_features = [
            'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
            'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin'
        ]
        self.categorical_features = [
            'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Drug'
        ]
        self.target_column = 'Status'
        
        # Models
        self.base_models: Dict = {}
        self.meta_model: Optional[LogisticRegression] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.decision_tree: Optional[DecisionTreeClassifier] = None
        
        # Class labels
        self.classes_ = ['C', 'CL', 'D']
        
        # Feature names after preprocessing
        self.feature_names_: List[str] = []
        
    def _create_preprocessor(self) -> ColumnTransformer:
        """Create preprocessing pipeline for features."""
        
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_features),
            ('cat', categorical_pipeline, self.categorical_features)
        ])
        
        return preprocessor
    
    def _create_base_models(self) -> Dict:
        """Create base models for the ensemble."""
        
        base_models = {
            'xgboost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            ),
            # The Interpreter Model - shallow for interpretability
            'decision_tree': DecisionTreeClassifier(
                max_depth=4,  # Keep shallow for interpretability!
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        }
        
        return base_models
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        
        # Separate features and target
        X = df[self.numerical_features + self.categorical_features].copy()
        y = df[self.target_column].copy()
        
        # Encode target
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_.tolist()
        
        # Fit preprocessor and transform
        self.preprocessor = self._create_preprocessor()
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        self._get_feature_names()
        
        return X_processed, y_encoded
    
    def _get_feature_names(self):
        """Extract feature names after preprocessing."""
        feature_names = []
        
        # Numerical features (unchanged names)
        feature_names.extend(self.numerical_features)
        
        # Categorical features (one-hot encoded)
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_features)
        feature_names.extend(cat_feature_names.tolist())
        
        self.feature_names_ = feature_names
    
    def train(self, data_path: str = "data/raw/cirrhosis.csv") -> Dict:
        """
        Train the stacked ensemble using cross-validation.
        
        Returns:
            Dict with training metrics and model info.
        """
        # Load data
        df = pd.read_csv(data_path)
        
        # Drop ID and N_Days (not features)
        df = df.drop(columns=['ID', 'N_Days'], errors='ignore')
        
        # Remove rows with missing target
        df = df.dropna(subset=[self.target_column])
        
        # Prepare data
        X, y = self._prepare_data(df)
        
        # Create base models
        self.base_models = self._create_base_models()
        
        # Cross-validation for stacking
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Generate out-of-fold predictions for meta-model training
        meta_features = np.zeros((X.shape[0], len(self.base_models) * len(self.classes_)))
        
        col_idx = 0
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            
            # Get out-of-fold probability predictions
            oof_proba = cross_val_predict(
                model, X, y, cv=cv, method='predict_proba'
            )
            
            # Store probabilities as meta-features
            meta_features[:, col_idx:col_idx + len(self.classes_)] = oof_proba
            col_idx += len(self.classes_)
            
            # Fit model on full data for inference
            model.fit(X, y)
        
        # Store decision tree reference
        self.decision_tree = self.base_models['decision_tree']
        
        # Train meta-model on out-of-fold predictions
        print("Training meta-model (Logistic Regression)...")
        self.meta_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial'
        )
        self.meta_model.fit(meta_features, y)
        
        # Calculate training accuracy
        final_predictions = self.meta_model.predict(meta_features)
        accuracy = (final_predictions == y).mean()
        
        # Save models
        self._save_models()
        
        return {
            'accuracy': accuracy,
            'n_samples': len(y),
            'n_features': X.shape[1],
            'classes': self.classes_,
            'base_models': list(self.base_models.keys())
        }
    
    def _save_models(self):
        """Save trained models to disk."""
        joblib.dump(self.base_models, self.model_dir / 'base_models.joblib')
        joblib.dump(self.meta_model, self.model_dir / 'meta_model.joblib')
        joblib.dump(self.preprocessor, self.model_dir / 'preprocessor.joblib')
        joblib.dump(self.label_encoder, self.model_dir / 'label_encoder.joblib')
        joblib.dump(self.feature_names_, self.model_dir / 'feature_names.joblib')
        print(f"Models saved to {self.model_dir}")
    
    def load_models(self):
        """Load trained models from disk."""
        self.base_models = joblib.load(self.model_dir / 'base_models.joblib')
        self.meta_model = joblib.load(self.model_dir / 'meta_model.joblib')
        self.preprocessor = joblib.load(self.model_dir / 'preprocessor.joblib')
        self.label_encoder = joblib.load(self.model_dir / 'label_encoder.joblib')
        self.feature_names_ = joblib.load(self.model_dir / 'feature_names.joblib')
        self.decision_tree = self.base_models['decision_tree']
        self.classes_ = self.label_encoder.classes_.tolist()
        print("Models loaded successfully")
    
    def _patient_to_dataframe(self, patient: PatientInput) -> pd.DataFrame:
        """Convert PatientInput to DataFrame for prediction."""
        data = {
            'Age': patient.age,
            'Sex': patient.sex.value,
            'Drug': patient.drug.value if patient.drug else None,
            'Ascites': patient.ascites.value,
            'Hepatomegaly': patient.hepatomegaly.value,
            'Spiders': patient.spiders.value,
            'Edema': patient.edema.value,
            'Bilirubin': patient.bilirubin,
            'Cholesterol': patient.cholesterol,
            'Albumin': patient.albumin,
            'Copper': patient.copper,
            'Alk_Phos': patient.alk_phos,
            'SGOT': patient.sgot,
            'Tryglicerides': patient.tryglicerides,
            'Platelets': patient.platelets,
            'Prothrombin': patient.prothrombin
        }
        return pd.DataFrame([data])
    
    def predict(self, patient: PatientInput) -> Dict:
        """
        Make prediction using the full stacked ensemble.
        
        Returns:
            Dict with prediction, probabilities, and base model outputs.
        """
        # Convert patient to DataFrame
        df = self._patient_to_dataframe(patient)
        
        # Preprocess
        X = self.preprocessor.transform(df[self.numerical_features + self.categorical_features])
        
        # Get predictions from all base models
        meta_features = []
        base_predictions = []
        
        for name, model in self.base_models.items():
            proba = model.predict_proba(X)[0]
            pred_idx = np.argmax(proba)
            
            meta_features.extend(proba)
            base_predictions.append(BaseModelPrediction(
                model_name=name,
                prediction=self.classes_[pred_idx],
                probability=float(proba[pred_idx]),
                confidence=float(np.max(proba) - np.partition(proba, -2)[-2])  # margin
            ))
        
        # Meta-model prediction
        meta_features = np.array(meta_features).reshape(1, -1)
        final_pred_idx = self.meta_model.predict(meta_features)[0]
        final_proba = self.meta_model.predict_proba(meta_features)[0]
        
        final_prediction = self.classes_[final_pred_idx]
        confidence = float(final_proba[final_pred_idx])
        
        # Determine risk level
        if final_prediction == 'D':
            risk_level = 'High'
        elif final_prediction == 'CL':
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'final_prediction': final_prediction,
            'confidence': confidence,
            'probabilities': {cls: float(p) for cls, p in zip(self.classes_, final_proba)},
            'base_model_predictions': base_predictions,
            'risk_level': risk_level
        }
    
    def get_decision_tree_path(self, patient: PatientInput) -> Dict:
        """
        Extract the decision path from the interpretable decision tree.
        
        Returns:
            Dict with decision rules and tree prediction.
        """
        # Convert and preprocess
        df = self._patient_to_dataframe(patient)
        X = self.preprocessor.transform(df[self.numerical_features + self.categorical_features])
        
        # Get decision path
        tree = self.decision_tree
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        
        node_indicator = tree.decision_path(X)
        node_indices = node_indicator.indices[
            node_indicator.indptr[0]:node_indicator.indptr[1]
        ]
        
        rules = []
        rule_texts = []
        
        for node_id in node_indices:
            # Skip leaf nodes
            if tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id]:
                continue
            
            feature_idx = feature[node_id]
            feature_name = self.feature_names_[feature_idx]
            thresh = threshold[node_id]
            
            # Check if sample goes left or right
            if X[0, feature_idx] <= thresh:
                operator = "<="
                direction = "left"
            else:
                operator = ">"
                direction = "right"
            
            rules.append({
                'feature': feature_name,
                'operator': operator,
                'threshold': round(float(thresh), 2),
                'direction': direction,
                'actual_value': round(float(X[0, feature_idx]), 2)
            })
            
            rule_texts.append(
                f"IF {feature_name} {operator} {thresh:.2f} "
                f"(actual: {X[0, feature_idx]:.2f}) → go {direction}"
            )
        
        # Tree prediction
        tree_pred_idx = tree.predict(X)[0]
        tree_prediction = self.classes_[tree_pred_idx]
        tree_proba = tree.predict_proba(X)[0]
        
        return {
            'decision_rules': rules,
            'rule_path': rule_texts,
            'tree_prediction': tree_prediction,
            'tree_confidence': float(tree_proba[tree_pred_idx])
        }
