#!/usr/bin/env python3
"""
Comprehensive Training Script for All Fraud Detection Models

This script provides a unified interface to train all four fraud detection models:
1. Website Fraud Detection
2. Mobile Risk Assessment
3. APK Malware Detection
4. Vision Brand Impersonation

Usage:
    python train_all_models.py --model all --data-dir ./data --output-dir ./artifacts
    python train_all_models.py --model website --data ./data/website_data.csv --output ./artifacts/website_fraud_models
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Import training modules
from train_website_fraud import WebsiteFraudTrainer
from train_mobile_risk import MobileRiskTrainer
from train_apk_malware import APKMalwareTrainer
from train_vision_brand import VisionBrandTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveTrainer:
    """
    Comprehensive trainer for all fraud detection models.
    
    This class orchestrates the training of all four models
    with proper error handling and progress tracking.
    """

    def __init__(self):
        """Initialize the comprehensive trainer."""
        self.trainers = {
            'website': WebsiteFraudTrainer(),
            'mobile': MobileRiskTrainer(),
            'apk': APKMalwareTrainer(),
            'vision': VisionBrandTrainer()
        }
        
        self.training_results = {}
    
    def train_website_model(self, data_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Train the website fraud detection model.
        
        Args:
            data_path: Path to website training data
            output_dir: Output directory for model files
            
        Returns:
            Dictionary containing training results
        """
        logger.info("Starting website fraud detection model training")
        
        try:
            trainer = self.trainers['website']
            trainer.run_training_pipeline(data_path, output_dir)
            
            result = {
                'status': 'success',
                'model_type': 'Website Fraud Detection',
                'output_dir': output_dir,
                'message': 'Website fraud detection model trained successfully'
            }
            
            logger.info("Website fraud detection model training completed")
            return result
            
        except Exception as e:
            error_msg = f"Website fraud detection model training failed: {e}"
            logger.error(error_msg)
            
            result = {
                'status': 'error',
                'model_type': 'Website Fraud Detection',
                'output_dir': output_dir,
                'message': error_msg
            }
            
            return result
    
    def train_mobile_model(self, data_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Train the mobile risk assessment model.
        
        Args:
            data_path: Path to mobile training data
            output_dir: Output directory for model files
            
        Returns:
            Dictionary containing training results
        """
        logger.info("Starting mobile risk assessment model training")
        
        try:
            trainer = self.trainers['mobile']
            trainer.run_training_pipeline(data_path, output_dir)
            
            result = {
                'status': 'success',
                'model_type': 'Mobile Risk Assessment',
                'output_dir': output_dir,
                'message': 'Mobile risk assessment model trained successfully'
            }
            
            logger.info("Mobile risk assessment model training completed")
            return result
            
        except Exception as e:
            error_msg = f"Mobile risk assessment model training failed: {e}"
            logger.error(error_msg)
            
            result = {
                'status': 'error',
                'model_type': 'Mobile Risk Assessment',
                'output_dir': output_dir,
                'message': error_msg
            }
            
            return result
    
    def train_apk_model(self, data_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Train the APK malware detection model.
        
        Args:
            data_path: Path to APK training data
            output_dir: Output directory for model files
            
        Returns:
            Dictionary containing training results
        """
        logger.info("Starting APK malware detection model training")
        
        try:
            trainer = self.trainers['apk']
            trainer.run_training_pipeline(data_path, output_dir)
            
            result = {
                'status': 'success',
                'model_type': 'APK Malware Detection',
                'output_dir': output_dir,
                'message': 'APK malware detection model trained successfully'
            }
            
            logger.info("APK malware detection model training completed")
            return result
            
        except Exception as e:
            error_msg = f"APK malware detection model training failed: {e}"
            logger.error(error_msg)
            
            result = {
                'status': 'error',
                'model_type': 'APK Malware Detection',
                'output_dir': output_dir,
                'message': error_msg
            }
            
            return result
    
    def train_vision_model(self, data_path: str, output_dir: str, coco_path: str = None) -> Dict[str, Any]:
        """
        Train the vision brand impersonation model.
        
        Args:
            data_path: Path to vision training data
            output_dir: Output directory for model files
            coco_path: Optional path to COCO annotation file
            
        Returns:
            Dictionary containing training results
        """
        logger.info("Starting vision brand impersonation model training")
        
        try:
            trainer = self.trainers['vision']
            trainer.run_training_pipeline(data_path, output_dir, coco_path)
            
            result = {
                'status': 'success',
                'model_type': 'Vision Brand Impersonation',
                'output_dir': output_dir,
                'message': 'Vision brand impersonation model trained successfully'
            }
            
            logger.info("Vision brand impersonation model training completed")
            return result
            
        except Exception as e:
            error_msg = f"Vision brand impersonation model training failed: {e}"
            logger.error(error_msg)
            
            result = {
                'status': 'error',
                'model_type': 'Vision Brand Impersonation',
                'output_dir': output_dir,
                'message': error_msg
            }
            
            return result
    
    def train_all_models(self, data_dir: str, output_dir: str, coco_path: str = None) -> Dict[str, Any]:
        """
        Train all four fraud detection models.
        
        Args:
            data_dir: Directory containing all training data files
            output_dir: Base output directory for all models
            coco_path: Optional path to COCO annotation file
            
        Returns:
            Dictionary containing results for all models
        """
        logger.info("Starting comprehensive training of all fraud detection models")
        
        # Define expected data files and output directories
        model_configs = {
            'website': {
                'data_file': os.path.join(data_dir, 'website_data.csv'),
                'output_dir': os.path.join(output_dir, 'website_fraud_models')
            },
            'mobile': {
                'data_file': os.path.join(data_dir, 'mobile_data.csv'),
                'output_dir': os.path.join(output_dir, 'mobile_risk_models')
            },
            'apk': {
                'data_file': os.path.join(data_dir, 'apk_data.csv'),
                'output_dir': os.path.join(output_dir, 'apk_malware_models')
            },
            'vision': {
                'data_file': os.path.join(data_dir, 'vision_data.csv'),
                'output_dir': os.path.join(output_dir, 'vision_brand_models')
            }
        }
        
        # Train each model
        results = {}
        
        for model_name, config in model_configs.items():
            logger.info(f"Training {model_name} model...")
            
            # Check if data file exists
            if not os.path.exists(config['data_file']):
                logger.warning(f"Data file not found: {config['data_file']}")
                results[model_name] = {
                    'status': 'skipped',
                    'model_type': config['output_dir'].split('/')[-1],
                    'output_dir': config['output_dir'],
                    'message': f'Data file not found: {config["data_file"]}'
                }
                continue
            
            # Train model based on type
            if model_name == 'website':
                results[model_name] = self.train_website_model(
                    config['data_file'], config['output_dir']
                )
            elif model_name == 'mobile':
                results[model_name] = self.train_mobile_model(
                    config['data_file'], config['output_dir']
                )
            elif model_name == 'apk':
                results[model_name] = self.train_apk_model(
                    config['data_file'], config['output_dir']
                )
            elif model_name == 'vision':
                results[model_name] = self.train_vision_model(
                    config['data_file'], config['output_dir'], coco_path
                )
        
        # Summary
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        total = len(results)
        
        logger.info(f"Training completed: {successful}/{total} models successful")
        
        return results
    
    def print_training_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a summary of training results.
        
        Args:
            results: Dictionary containing training results
        """
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        for model_name, result in results.items():
            status_icon = "✓" if result['status'] == 'success' else "✗" if result['status'] == 'error' else "-"
            print(f"{status_icon} {result['model_type']}: {result['status'].upper()}")
            
            if result['status'] == 'success':
                print(f"   Output: {result['output_dir']}")
            elif result['status'] == 'error':
                print(f"   Error: {result['message']}")
            elif result['status'] == 'skipped':
                print(f"   Reason: {result['message']}")
        
        print("="*80)
        
        # Count results
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        errors = sum(1 for r in results.values() if r['status'] == 'error')
        skipped = sum(1 for r in results.values() if r['status'] == 'skipped')
        
        print(f"Total: {len(results)} | Successful: {successful} | Errors: {errors} | Skipped: {skipped}")
        print("="*80)


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train All Fraud Detection Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models
  python train_all_models.py --model all --data-dir ./data --output-dir ./artifacts
  
  # Train specific model
  python train_all_models.py --model website --data ./data/website_data.csv --output ./artifacts/website_fraud_models
  
  # Train vision model with COCO annotations
  python train_all_models.py --model vision --data ./data/vision_data.csv --output ./artifacts/vision_brand_models --coco ./data/annotations.json
        """
    )
    
    parser.add_argument(
        '--model', 
        choices=['all', 'website', 'mobile', 'apk', 'vision'],
        default='all',
        help='Model to train (default: all)'
    )
    
    parser.add_argument(
        '--data', 
        help='Path to training data file (for single model training)'
    )
    
    parser.add_argument(
        '--data-dir', 
        help='Directory containing training data files (for all models)'
    )
    
    parser.add_argument(
        '--output', 
        help='Output directory for model files (for single model training)'
    )
    
    parser.add_argument(
        '--output-dir', 
        default='./artifacts',
        help='Base output directory for all models (default: ./artifacts)'
    )
    
    parser.add_argument(
        '--coco', 
        help='Path to COCO annotation file (for vision model)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model != 'all':
        if not args.data:
            parser.error(f"--data is required when training {args.model} model")
        if not args.output:
            parser.error(f"--output is required when training {args.model} model")
    else:
        if not args.data_dir:
            parser.error("--data-dir is required when training all models")
    
    # Initialize trainer
    trainer = ComprehensiveTrainer()
    
    try:
        if args.model == 'all':
            # Train all models
            results = trainer.train_all_models(args.data_dir, args.output_dir, args.coco)
            trainer.print_training_summary(results)
        else:
            # Train single model
            if args.model == 'website':
                result = trainer.train_website_model(args.data, args.output)
            elif args.model == 'mobile':
                result = trainer.train_mobile_model(args.data, args.output)
            elif args.model == 'apk':
                result = trainer.train_apk_model(args.data, args.output)
            elif args.model == 'vision':
                result = trainer.train_vision_model(args.data, args.output, args.coco)
            
            # Print result
            if result['status'] == 'success':
                logger.info(f"{result['model_type']} training completed successfully")
                logger.info(f"Model saved to: {result['output_dir']}")
            else:
                logger.error(f"{result['model_type']} training failed: {result['message']}")
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
