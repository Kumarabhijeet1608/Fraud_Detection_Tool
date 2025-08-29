#!/usr/bin/env python3
"""
Website Fraud Detection Model Training Script

This script trains a logistic regression model for website fraud detection
using URL features and TF-IDF text analysis.

Features:
- URL structure analysis
- Domain extraction and analysis
- TF-IDF text processing
- Model calibration and validation
"""

import pandas as pd
import numpy as np
import re
import json
import pickle
import logging
from typing import Dict, List, Tuple, Any
from urllib.parse import urlparse, parse_qs
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import tldextract
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WebsiteFraudTrainer:
    """
    Trainer class for website fraud detection model.
    
    This class handles data loading, feature extraction, model training,
    and evaluation for website fraud detection.
    """

    def __init__(self):
        """Initialize the trainer with default parameters."""
        self.model = None
        self.tfidf_vectorizer = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load training data from file.
        
        Args:
            data_path: Path to the training data file
            
        Returns:
            Loaded DataFrame with URL and label columns
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
    
    def extract_url_features(self, url: str) -> Dict[str, Any]:
        """
        Extract features from a URL for fraud detection.
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary containing extracted features
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            query = parsed.query.lower()
            
            # Extract domain components
            domain_parts = tldextract.extract(url)
            
            # Basic URL features
            features = {
                'url_length': len(url),
                'domain_length': len(domain),
                'path_length': len(path),
                'query_length': len(query),
                'domain_dots': domain.count('.'),
                'path_slashes': path.count('/'),
                'query_params': query.count('&'),
                'digit_count': sum(c.isdigit() for c in url),
                'uppercase_count': sum(c.isupper() for c in url),
                'special_char_count': sum(c in '!@#$%^&*()' for c in url),
                'subdomain_count': len(domain_parts.subdomain.split('.')) if domain_parts.subdomain else 0,
                'tld_length': len(domain_parts.suffix),
                'has_https': 1 if parsed.scheme == 'https' else 0,
                'has_www': 1 if 'www.' in domain else 0,
                'is_ip': 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', domain) else 0
            }
            
            # Text features for TF-IDF
            text_features = f"{domain} {path} {query}"
            
            return {
                'numeric_features': features,
                'text_features': text_features
            }
            
        except Exception as e:
            logger.error(f"Error extracting features from URL {url}: {e}")
            # Return default features on error
            return {
                'numeric_features': {k: 0 for k in range(16)},
                'text_features': ''
            }
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for training.
        
        Args:
            df: DataFrame with URL and label columns
            
        Returns:
            Tuple of (X_features, y_labels, feature_names)
        """
        logger.info("Preparing features for training")
        
        # Extract features from URLs
        feature_data = []
        text_data = []
        
        for url in df['url']:
            features = self.extract_url_features(url)
            feature_data.append(list(features['numeric_features'].values()))
            text_data.append(features['text_features'])
        
        # Convert to numpy arrays
        X_numeric = np.array(feature_data)
        X_text = np.array(text_data)
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        X_tfidf = self.tfidf_vectorizer.fit_transform(X_text)
        
        # Combine numeric and text features
        X_combined = np.hstack([X_numeric, X_tfidf.toarray()])
        
        # Prepare labels
        y = df['label'].values
        
        # Store feature names
        numeric_feature_names = list(features['numeric_features'].keys())
        text_feature_names = [f'text_{i}' for i in range(X_tfidf.shape[1])]
        self.feature_names = numeric_feature_names + text_feature_names
        
        logger.info(f"Prepared {X_combined.shape[1]} features")
        return X_combined, y, self.feature_names
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the fraud detection model.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        logger.info("Training fraud detection model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numeric features (first 16 columns)
        self.scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[:, :16] = self.scaler.fit_transform(X_train[:, :16])
        X_test_scaled[:, :16] = self.scaler.transform(X_test[:, :16])
        
        # Train base model
        base_model = LogisticRegression(
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
            base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Calibrate model for better probability estimates
        self.model = CalibratedClassifierCV(
            best_model, cv=5, method='isotonic'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        auc = roc_auc_score(y_test, y_prob)
        
        logger.info(f"Model training completed")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Test AUC: {auc:.4f}")
        
        # Print detailed classification report
        logger.info("Classification Report:")
        logger.info(classification_report(y_test, y_pred))
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the trained model and preprocessors.
        
        Args:
            output_dir: Directory to save model files
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, 'website_fraud_lr_calibrated.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save TF-IDF vectorizer
        tfidf_path = os.path.join(output_dir, 'website_fraud_lr_tfidf.pkl')
        with open(tfidf_path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'website_fraud_lr_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names
        features_path = os.path.join(output_dir, 'website_fraud_lr_features.json')
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        logger.info(f"Model saved to {output_dir}")
    
    def run_training_pipeline(self, data_path: str, output_dir: str) -> None:
        """
        Run the complete training pipeline.
        
        Args:
            data_path: Path to training data
            output_dir: Directory to save trained model
        """
        logger.info("Starting website fraud detection training pipeline")
        
        try:
            # Load data
            df = self.load_data(data_path)
            
            # Prepare features
            X, y, feature_names = self.prepare_features(df)
            
            # Train model
            self.train_model(X, y)
            
            # Save model
            self.save_model(output_dir)
            
            logger.info("Training pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function to run the training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Website Fraud Detection Model')
    parser.add_argument('--data', required=True, help='Path to training data file')
    parser.add_argument('--output', required=True, help='Output directory for model files')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = WebsiteFraudTrainer()
    
    # Run training pipeline
    trainer.run_training_pipeline(args.data, args.output)


if __name__ == "__main__":
    main()
