#!/usr/bin/env python3
"""
Mobile Risk Assessment Model Training Script

This script trains ensemble models for mobile app risk assessment
using app reviews and PlayStore metadata.

Features:
- Text sentiment analysis from reviews
- PlayStore metadata analysis
- Ensemble fusion of multiple models
- Cross-validation and hyperparameter tuning
"""

import pandas as pd
import numpy as np
import json
import pickle
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MobileRiskTrainer:
    """
    Trainer class for mobile risk assessment models.
    
    This class handles training of separate models for reviews and metadata,
    then combines them using an ensemble fusion approach.
    """

    def __init__(self):
        """Initialize the trainer with default parameters."""
        self.reviews_model = None
        self.metadata_model = None
        self.fusion_model = None
        self.tfidf_vectorizer = None
        self.metadata_preprocessor = None
        self.label_encoder = None
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load training data from file.
        
        Args:
            data_path: Path to the training data file
            
        Returns:
            Loaded DataFrame with reviews, metadata, and label columns
        """
        try:
            logger.info(f"Loading data from {data_path}")
            
            # Try different file formats
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                raise ValueError("Unsupported file format")
            
            logger.info(f"Loaded {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def prepare_reviews_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare text features from app reviews.
        
        Args:
            df: DataFrame containing reviews and labels
            
        Returns:
            Tuple of (X_reviews, y_labels)
        """
        logger.info("Preparing reviews features")
        
        # Extract reviews text
        reviews = df['reviews'].fillna('').astype(str)
        labels = df['label'].values
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        X_reviews = self.tfidf_vectorizer.fit_transform(reviews)
        
        logger.info(f"Prepared {X_reviews.shape[1]} review features")
        return X_reviews, labels
    
    def prepare_metadata_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features from PlayStore metadata.
        
        Args:
            df: DataFrame containing metadata and labels
            
        Returns:
            Tuple of (X_metadata, y_labels)
        """
        logger.info("Preparing metadata features")
        
        # Select metadata columns
        metadata_columns = [
            'rating', 'reviews_count', 'downloads', 'price', 'size',
            'content_rating', 'category', 'last_updated', 'min_android_version'
        ]
        
        # Filter available columns
        available_columns = [col for col in metadata_columns if col in df.columns]
        
        if not available_columns:
            raise ValueError("No metadata columns found in dataset")
        
        # Prepare metadata features
        metadata_df = df[available_columns].copy()
        
        # Handle categorical variables
        categorical_columns = metadata_df.select_dtypes(include=['object']).columns
        numerical_columns = metadata_df.select_dtypes(include=['number']).columns
        
        # Create preprocessing pipeline
        preprocessors = []
        
        if len(numerical_columns) > 0:
            numerical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            preprocessors.append(('num', numerical_transformer, numerical_columns))
        
        if len(categorical_columns) > 0:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', LabelEncoder())
            ])
            preprocessors.append(('cat', categorical_transformer, categorical_columns))
        
        # Create column transformer
        self.metadata_preprocessor = ColumnTransformer(
            transformers=preprocessors,
            remainder='drop'
        )
        
        # Transform metadata
        X_metadata = self.metadata_preprocessor.fit_transform(metadata_df)
        labels = df['label'].values
        
        logger.info(f"Prepared {X_metadata.shape[1]} metadata features")
        return X_metadata, labels
    
    def train_reviews_model(self, X_reviews: np.ndarray, y: np.ndarray) -> None:
        """
        Train the reviews sentiment analysis model.
        
        Args:
            X_reviews: Review features matrix
            y: Target labels
        """
        logger.info("Training reviews sentiment model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_reviews, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.reviews_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2']
        }
        
        grid_search = GridSearchCV(
            self.reviews_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.reviews_model = grid_search.best_estimator_
        
        # Evaluate model
        y_pred = self.reviews_model.predict(X_test)
        y_prob = self.reviews_model.predict_proba(X_test)[:, 1]
        
        accuracy = (y_pred == y_test).mean()
        auc = roc_auc_score(y_test, y_prob)
        
        logger.info(f"Reviews model training completed")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Test AUC: {auc:.4f}")
    
    def train_metadata_model(self, X_metadata: np.ndarray, y: np.ndarray) -> None:
        """
        Train the metadata analysis model.
        
        Args:
            X_metadata: Metadata features matrix
            y: Target labels
        """
        logger.info("Training metadata analysis model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_metadata, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.metadata_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            self.metadata_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.metadata_model = grid_search.best_estimator_
        
        # Evaluate model
        y_pred = self.metadata_model.predict(X_test)
        y_prob = self.metadata_model.predict_proba(X_test)[:, 1]
        
        accuracy = (y_pred == y_test).mean()
        auc = roc_auc_score(y_test, y_prob)
        
        logger.info(f"Metadata model training completed")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Test AUC: {auc:.4f}")
    
    def train_fusion_model(self, X_reviews: np.ndarray, X_metadata: np.ndarray, y: np.ndarray) -> None:
        """
        Train the ensemble fusion model.
        
        Args:
            X_reviews: Review features matrix
            X_metadata: Metadata features matrix
            y: Target labels
        """
        logger.info("Training ensemble fusion model")
        
        # Get predictions from individual models
        reviews_probs = self.reviews_model.predict_proba(X_reviews)[:, 1]
        metadata_probs = self.metadata_model.predict_proba(X_metadata)[:, 1]
        
        # Combine predictions
        X_fusion = np.column_stack([reviews_probs, metadata_probs])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_fusion, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train fusion model
        self.fusion_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4]
        }
        
        grid_search = GridSearchCV(
            self.fusion_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.fusion_model = grid_search.best_estimator_
        
        # Evaluate model
        y_pred = self.fusion_model.predict(X_test)
        y_prob = self.fusion_model.predict_proba(X_test)[:, 1]
        
        accuracy = (y_pred == y_test).mean()
        auc = roc_auc_score(y_test, y_prob)
        
        logger.info(f"Fusion model training completed")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Test AUC: {auc:.4f}")
    
    def save_models(self, output_dir: str) -> None:
        """
        Save all trained models and preprocessors.
        
        Args:
            output_dir: Directory to save model files
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save reviews model
        reviews_path = os.path.join(output_dir, 'mobile_reviews_model.pkl')
        with open(reviews_path, 'wb') as f:
            pickle.dump(self.reviews_model, f)
        
        # Save metadata model
        metadata_path = os.path.join(output_dir, 'mobile_playstore_model.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata_model, f)
        
        # Save fusion model
        fusion_path = os.path.join(output_dir, 'mobile_fusion_model.pkl')
        with open(fusion_path, 'wb') as f:
            pickle.dump(self.fusion_model, f)
        
        # Save TF-IDF vectorizer
        tfidf_path = os.path.join(output_dir, 'mobile_reviews_tfidf.pkl')
        with open(tfidf_path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # Save metadata preprocessor
        preprocessor_path = os.path.join(output_dir, 'mobile_playstore_preprocessor.pkl')
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.metadata_preprocessor, f)
        
        logger.info(f"All models saved to {output_dir}")
    
    def run_training_pipeline(self, data_path: str, output_dir: str) -> None:
        """
        Run the complete training pipeline.
        
        Args:
            data_path: Path to training data
            output_dir: Directory to save trained models
        """
        logger.info("Starting mobile risk assessment training pipeline")
        
        try:
            # Load data
            df = self.load_data(data_path)
            
            # Prepare features
            X_reviews, y_reviews = self.prepare_reviews_features(df)
            X_metadata, y_metadata = self.prepare_metadata_features(df)
            
            # Train individual models
            self.train_reviews_model(X_reviews, y_reviews)
            self.train_metadata_model(X_metadata, y_metadata)
            
            # Train fusion model
            self.train_fusion_model(X_reviews, X_metadata, y_reviews)
            
            # Save models
            self.save_models(output_dir)
            
            logger.info("Training pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function to run the training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Mobile Risk Assessment Models')
    parser.add_argument('--data', required=True, help='Path to training data file')
    parser.add_argument('--output', required=True, help='Output directory for model files')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MobileRiskTrainer()
    
    # Run training pipeline
    trainer.run_training_pipeline(args.data, args.output)


if __name__ == "__main__":
    main()
