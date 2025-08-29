#!/usr/bin/env python3
"""
Vision Brand Impersonation Detection Model Training Script

This script trains ExtraTrees models for detecting brand impersonation
using image features and COCO annotation data.

Features:
- Image feature extraction
- COCO annotation processing
- Bounding box analysis
- Brand text detection
- Multi-modal feature fusion
"""

import pandas as pd
import numpy as np
import json
import pickle
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisionBrandTrainer:
    """
    Trainer class for vision brand impersonation detection.
    
    This class handles training of models for detecting brand impersonation
    using image features and COCO annotations.
    """

    def __init__(self):
        """Initialize the trainer with default parameters."""
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        self.feature_names = None
        self.selected_features = None
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load training data from file.
        
        Args:
            data_path: Path to the training data file
            
        Returns:
            Loaded DataFrame with image features and labels
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
            
            logger.info(f"Loaded {len(df)} image samples")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def load_coco_annotations(self, coco_path: str) -> Dict[str, Any]:
        """
        Load COCO format annotations for additional context.
        
        Args:
            coco_path: Path to COCO annotation file
            
        Returns:
            Dictionary containing COCO annotations
        """
        try:
            logger.info(f"Loading COCO annotations from {coco_path}")
            
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)
            
            logger.info(f"Loaded COCO data with {len(coco_data.get('annotations', []))} annotations")
            return coco_data
            
        except Exception as e:
            logger.error(f"Failed to load COCO annotations: {e}")
            return {}
    
    def prepare_features(self, df: pd.DataFrame, coco_data: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for training.
        
        Args:
            df: DataFrame with image features and labels
            coco_data: Optional COCO annotation data
            
        Returns:
            Tuple of (X_features, y_labels, feature_names)
        """
        logger.info("Preparing vision features for training")
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col != 'label']
        X = df[feature_columns].values
        y = df['label'].values
        
        # Store feature names
        self.feature_names = feature_columns
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Convert labels to numeric if needed
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        
        # Add COCO-based features if available
        if coco_data:
            X = self._add_coco_features(X, df, coco_data)
        
        logger.info(f"Prepared {X.shape[1]} features from {len(feature_columns)} columns")
        return X, y, self.feature_names
    
    def _add_coco_features(self, X: np.ndarray, df: pd.DataFrame, coco_data: Dict[str, Any]) -> np.ndarray:
        """
        Add COCO annotation-based features to the feature matrix.
        
        Args:
            X: Original feature matrix
            df: Original DataFrame
            coco_data: COCO annotation data
            
        Returns:
            Enhanced feature matrix with COCO features
        """
        logger.info("Adding COCO annotation features")
        
        # Extract COCO features
        coco_features = []
        
        for idx, row in df.iterrows():
            # Basic COCO features (placeholder - adjust based on actual data structure)
            image_id = row.get('image_id', idx)
            
            # Count annotations for this image
            annotations = [ann for ann in coco_data.get('annotations', []) 
                         if ann.get('image_id') == image_id]
            
            # Extract bounding box features
            bbox_count = len(annotations)
            total_area = sum(ann.get('area', 0) for ann in annotations)
            avg_area = total_area / max(bbox_count, 1)
            
            # Category distribution
            categories = [ann.get('category_id', 0) for ann in annotations]
            unique_categories = len(set(categories))
            
            # Combine features
            coco_feature_vector = [
                bbox_count,
                total_area,
                avg_area,
                unique_categories,
                len([c for c in categories if c == 1]),  # Assuming category 1 is brand-related
                len([c for c in categories if c == 2])   # Assuming category 2 is text-related
            ]
            
            coco_features.append(coco_feature_vector)
        
        # Convert to numpy array
        coco_features_array = np.array(coco_features)
        
        # Combine with original features
        X_enhanced = np.hstack([X, coco_features_array])
        
        # Update feature names
        coco_feature_names = [
            'bbox_count', 'total_area', 'avg_area', 
            'unique_categories', 'brand_objects', 'text_objects'
        ]
        self.feature_names.extend(coco_feature_names)
        
        logger.info(f"Added {len(coco_feature_names)} COCO features")
        return X_enhanced
    
    def select_features(self, X: np.ndarray, y: np.ndarray, k: int = 200) -> np.ndarray:
        """
        Select the most important features using statistical tests.
        
        Args:
            X: Feature matrix
            y: Target labels
            k: Number of features to select
            
        Returns:
            Feature matrix with selected features
        """
        logger.info(f"Selecting top {k} features")
        
        # Use F-test for feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support()
        self.selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_indices[i]]
        
        logger.info(f"Selected {X_selected.shape[1]} features")
        return X_selected
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the vision brand impersonation detection model.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        logger.info("Training vision brand impersonation detection model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train base model
        base_model = ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
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
        
        # Feature importance analysis
        feature_importance = self.model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
        
        logger.info("Top 10 most important features:")
        for idx in reversed(top_features_idx):
            feature_name = self.selected_features[idx] if self.selected_features else f"feature_{idx}"
            importance = feature_importance[idx]
            logger.info(f"  {feature_name}: {importance:.4f}")
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the trained model and preprocessors.
        
        Args:
            output_dir: Directory to save model files
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, 'vision_brand_et_coco.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'vision_brand_et_coco_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature selector
        selector_path = os.path.join(output_dir, 'vision_brand_et_coco_selector.pkl')
        with open(selector_path, 'wb') as f:
            pickle.dump(self.feature_selector, f)
        
        # Save feature names
        features_path = os.path.join(output_dir, 'vision_brand_et_coco_features.json')
        with open(features_path, 'w') as f:
            json.dump(self.selected_features, f)
        
        # Save label encoder if used
        if self.label_encoder:
            encoder_path = os.path.join(output_dir, 'vision_brand_et_coco_encoder.pkl')
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        logger.info(f"Model saved to {output_dir}")
    
    def run_training_pipeline(self, data_path: str, output_dir: str, coco_path: str = None, feature_count: int = 200) -> None:
        """
        Run the complete training pipeline.
        
        Args:
            data_path: Path to training data
            output_dir: Directory to save trained model
            coco_path: Optional path to COCO annotation file
            feature_count: Number of features to select
        """
        logger.info("Starting vision brand impersonation training pipeline")
        
        try:
            # Load data
            df = self.load_data(data_path)
            
            # Load COCO annotations if provided
            coco_data = {}
            if coco_path:
                coco_data = self.load_coco_annotations(coco_path)
            
            # Prepare features
            X, y, feature_names = self.prepare_features(df, coco_data)
            
            # Select features
            X_selected = self.select_features(X, y, feature_count)
            
            # Train model
            self.train_model(X_selected, y)
            
            # Save model
            self.save_model(output_dir)
            
            logger.info("Training pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function to run the training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Vision Brand Impersonation Detection Model')
    parser.add_argument('--data', required=True, help='Path to training data file')
    parser.add_argument('--output', required=True, help='Output directory for model files')
    parser.add_argument('--coco', help='Path to COCO annotation file (optional)')
    parser.add_argument('--features', type=int, default=200, help='Number of features to select')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = VisionBrandTrainer()
    
    # Run training pipeline
    trainer.run_training_pipeline(args.data, args.output, args.coco, args.features)


if __name__ == "__main__":
    main()
