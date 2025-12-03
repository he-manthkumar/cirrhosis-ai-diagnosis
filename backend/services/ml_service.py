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

from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
    
    def __init__(self, model_dir: str = "backend/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature columns (matching preprocessed data)
        self.numerical_features = [
            'Age_Years', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
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
    
    def train(self, data_path: str = "data/raw/cirrhosis.csv", test_size: float = 0.2) -> Dict:
        """
        Train the stacked ensemble with proper train/test split.
        
        Args:
            data_path: Path to the dataset
            test_size: Fraction of data to hold out for testing (default: 20%)
        
        Returns:
            Dict with training AND test metrics for honest evaluation.
        """
        # Load data
        df = pd.read_csv(data_path)
        
        # Drop ID and N_Days (not features)
        df = df.drop(columns=['ID', 'N_Days'], errors='ignore')
        
        # Remove rows with missing target
        df = df.dropna(subset=[self.target_column])
        
        # Prepare data
        X, y = self._prepare_data(df)
        
        # ============================================================
        # CRITICAL: Proper Train/Test Split BEFORE any training!
        # ============================================================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training set: {len(y_train)} samples ({100*(1-test_size):.0f}%)")
        print(f"   Test set: {len(y_test)} samples ({100*test_size:.0f}%)")
        print(f"   (Test set is NEVER seen during training)\n")
        
        # Save test indices for later evaluation
        self.test_indices_ = {
            'X_test': X_test,
            'y_test': y_test
        }
        
        # Create base models
        self.base_models = self._create_base_models()
        
        # Cross-validation for stacking (ON TRAINING DATA ONLY!)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Generate out-of-fold predictions for meta-model training
        meta_features_train = np.zeros((X_train.shape[0], len(self.base_models) * len(self.classes_)))
        
        col_idx = 0
        train_metrics = {}
        
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            
            # Get out-of-fold probability predictions (on training data only)
            oof_proba = cross_val_predict(
                model, X_train, y_train, cv=cv, method='predict_proba'
            )
            
            # Store probabilities as meta-features
            meta_features_train[:, col_idx:col_idx + len(self.classes_)] = oof_proba
            col_idx += len(self.classes_)
            
            # Fit model on TRAINING data only (not full data!)
            model.fit(X_train, y_train)
        
        # Store decision tree reference
        self.decision_tree = self.base_models['decision_tree']
        
        # Train meta-model on out-of-fold predictions
        print("Training meta-model (Logistic Regression)...")
        self.meta_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        self.meta_model.fit(meta_features_train, y_train)
        
        # ============================================================
        # Evaluate on HELD-OUT TEST SET (honest evaluation!)
        # ============================================================
        print("\n🔍 Evaluating on held-out test set...")
        
        # Generate test meta-features
        meta_features_test = np.zeros((X_test.shape[0], len(self.base_models) * len(self.classes_)))
        col_idx = 0
        test_base_metrics = {}
        
        for name, model in self.base_models.items():
            # Predict on test set
            test_proba = model.predict_proba(X_test)
            test_pred = model.predict(X_test)
            
            meta_features_test[:, col_idx:col_idx + len(self.classes_)] = test_proba
            col_idx += len(self.classes_)
            
            # Calculate test metrics for each base model
            test_base_metrics[name] = {
                'accuracy': accuracy_score(y_test, test_pred),
                'f1': f1_score(y_test, test_pred, average='weighted')
            }
        
        # Final ensemble prediction on test set
        test_predictions = self.meta_model.predict(meta_features_test)
        test_proba_ensemble = self.meta_model.predict_proba(meta_features_test)
        
        # Calculate honest test metrics
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_precision = precision_score(y_test, test_predictions, average='weighted')
        test_recall = recall_score(y_test, test_predictions, average='weighted')
        test_f1 = f1_score(y_test, test_predictions, average='weighted')
        test_roc_auc = roc_auc_score(y_test, test_proba_ensemble, multi_class='ovr', average='weighted')
        
        # Save models
        self._save_models()
        
        # Save test data for notebook evaluation
        joblib.dump(self.test_indices_, self.model_dir / 'test_data.pkl')
        
        return {
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'n_features': X_train.shape[1],
            'classes': self.classes_,
            'base_models': list(self.base_models.keys()),
            # Honest test metrics
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_roc_auc': test_roc_auc,
            'test_base_metrics': test_base_metrics
        }
    
    def _save_models(self):
        """Save trained models to disk as .pkl files."""
        joblib.dump(self.base_models, self.model_dir / 'base_models.pkl')
        joblib.dump(self.meta_model, self.model_dir / 'meta_model.pkl')
        joblib.dump(self.preprocessor, self.model_dir / 'preprocessor.pkl')
        joblib.dump(self.label_encoder, self.model_dir / 'label_encoder.pkl')
        joblib.dump(self.feature_names_, self.model_dir / 'feature_names.pkl')
        print(f"Models saved to {self.model_dir}")
    
    def load_models(self):
        """Load trained models from disk."""
        self.base_models = joblib.load(self.model_dir / 'base_models.pkl')
        self.meta_model = joblib.load(self.model_dir / 'meta_model.pkl')
        self.preprocessor = joblib.load(self.model_dir / 'preprocessor.pkl')
        self.label_encoder = joblib.load(self.model_dir / 'label_encoder.pkl')
        self.feature_names_ = joblib.load(self.model_dir / 'feature_names.pkl')
        self.decision_tree = self.base_models['decision_tree']
        self.classes_ = self.label_encoder.classes_.tolist()
        print("Models loaded successfully")
    
    def _patient_to_dataframe(self, patient: PatientInput) -> pd.DataFrame:
        """Convert PatientInput to DataFrame for prediction."""
        data = {
            'Age_Years': patient.age,  # Map age to Age_Years
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
