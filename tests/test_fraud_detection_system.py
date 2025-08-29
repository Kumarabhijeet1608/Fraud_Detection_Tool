#!/usr/bin/env python3
"""
Tests for the Fraud Detection System

This module contains basic tests for the main fraud detection system.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.fraud_detection_system import FraudDetectionSystem


class TestFraudDetectionSystem(unittest.TestCase):
    """Test cases for the FraudDetectionSystem class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the model loading to avoid file dependencies
        with patch('builtins.open'), \
             patch('pickle.load'), \
             patch('json.load'):
            self.fraud_system = FraudDetectionSystem()

    def test_system_initialization(self):
        """Test that the system initializes correctly."""
        self.assertIsInstance(self.fraud_system, FraudDetectionSystem)
        self.assertIsInstance(self.fraud_system.models, dict)
        self.assertIsInstance(self.fraud_system.preprocessors, dict)
        self.assertIsInstance(self.fraud_system.model_info, dict)

    def test_unwrap_estimator(self):
        """Test the _unwrap_estimator method."""
        # Test with dict wrapper
        wrapped_estimator = {'estimator': 'test_model'}
        result = self.fraud_system._unwrap_estimator(wrapped_estimator)
        self.assertEqual(result, 'test_model')

        # Test with direct estimator
        direct_estimator = 'test_model'
        result = self.fraud_system._unwrap_estimator(direct_estimator)
        self.assertEqual(result, 'test_model')

    def test_extract_website_features(self):
        """Test website feature extraction."""
        url = "https://example.com/path?param=value"
        features = self.fraud_system._extract_website_features(url)
        
        self.assertIn('text', features)
        self.assertIn('numeric', features)
        self.assertEqual(len(features['numeric']), 10)
        self.assertIsInstance(features['text'], str)
        self.assertIsInstance(features['numeric'], list)

    def test_extract_website_features_invalid_url(self):
        """Test website feature extraction with invalid URL."""
        features = self.fraud_system._extract_website_features("invalid://url")
        self.assertIn('text', features)
        self.assertIn('numeric', features)
        self.assertEqual(len(features['numeric']), 10)

    def test_get_system_status(self):
        """Test system status retrieval."""
        status = self.fraud_system.get_system_status()
        
        self.assertIn('status', status)
        self.assertIn('total_models', status)
        self.assertIn('models', status)
        self.assertIn('model_info', status)
        self.assertIsInstance(status['total_models'], int)
        self.assertIsInstance(status['models'], list)

    def test_get_model_info(self):
        """Test model information retrieval."""
        # Test with existing model
        info = self.fraud_system.get_model_info('website_fraud')
        if info:
            self.assertIsInstance(info, dict)
        
        # Test with non-existing model
        info = self.fraud_system.get_model_info('non_existing_model')
        self.assertIsNone(info)


class TestFraudDetectionSystemIntegration(unittest.TestCase):
    """Integration tests for the Fraud Detection System."""

    @patch('src.models.fraud_detection_system.FraudDetectionSystem._load_website_fraud_model')
    @patch('src.models.fraud_detection_system.FraudDetectionSystem._load_mobile_risk_model')
    @patch('src.models.fraud_detection_system.FraudDetectionSystem._load_apk_malware_model')
    @patch('src.models.fraud_detection_system.FraudDetectionSystem._load_vision_brand_model')
    def test_model_loading_integration(self, mock_vision, mock_apk, mock_mobile, mock_website):
        """Test that all models are loaded during initialization."""
        fraud_system = FraudDetectionSystem()
        
        mock_website.assert_called_once()
        mock_mobile.assert_called_once()
        mock_apk.assert_called_once()
        mock_vision.assert_called_once()


if __name__ == '__main__':
    unittest.main()
