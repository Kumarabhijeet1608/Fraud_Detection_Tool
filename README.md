# Multi-Modal Fraud Detection System

A comprehensive fraud detection system that combines multiple specialized models to detect fraud across different domains including websites, mobile applications, APK files, and visual content.

## Overview

This system provides production-ready fraud detection capabilities through four specialized models:

- **Website Fraud Detection**: URL analysis and phishing detection (95.2% accuracy)
- **Mobile Risk Assessment**: App store metadata and review analysis (87.3% accuracy)
- **APK Malware Detection**: Android application security analysis (91.7% accuracy)
- **Vision Brand Impersonation**: Logo detection and brand verification (45.5% accuracy)

## Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd fraud-detection-tool

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models.fraud_detection_system import FraudDetectionSystem

# Initialize the system
fraud_system = FraudDetectionSystem()

# Website fraud detection
result = fraud_system.predict_website_fraud("http://example.com")

# Mobile risk assessment
result = fraud_system.predict_mobile_risk(reviews_text, playstore_features)

# APK malware detection
result = fraud_system.predict_apk_malware(apk_features)

# Vision brand detection
result = fraud_system.predict_vision_brand(image_features)
```

### Command Line Interface

```bash
# Check system status
python src/cli.py status

# Website fraud detection
python src/cli.py website-fraud "http://example.com"

# Mobile risk assessment
python src/cli.py mobile-risk "Great app!" --metadata '{"rating": 4.5, "downloads": 1000}'

# APK malware detection
python src/cli.py apk-malware --features-file features.csv

# Vision brand detection
python src/cli.py vision-brand --features-file image_features.csv
```

## System Architecture

### Model 1: Website Fraud Detection
- **Algorithm**: Logistic Regression with TF-IDF
- **Features**: URL text analysis, domain extraction, eTLD+1 splitting
- **Performance**: 95.2% validation accuracy
- **Use Case**: Phishing website detection, malicious URL identification

### Model 2: Mobile Risk Fusion
- **Algorithm**: Ensemble fusion (Reviews + PlayStore metadata)
- **Features**: Text sentiment, app permissions, ratings, downloads
- **Performance**: 87.3% validation accuracy
- **Use Case**: Mobile app risk assessment, fraud app detection

### Model 3: APK Malware Detection
- **Algorithm**: Random Forest with feature selection
- **Features**: API signatures, permissions, static analysis
- **Performance**: 91.7% validation accuracy
- **Use Case**: Android malware detection, suspicious app identification

### Model 4: Vision Brand Impersonation
- **Algorithm**: ExtraTrees with COCO integration
- **Features**: Bounding boxes, image metadata, brand text analysis
- **Performance**: 45.5% validation accuracy
- **Use Case**: Logo detection, brand impersonation identification

## Project Structure

```
fraud-detection-tool/
├── src/                           # Source code
│   ├── models/                    # Model implementations
│   ├── cli.py                     # Command line interface
│   └── utils/                     # Utility functions
├── artifacts/                     # Trained models and artifacts
├── data/                          # Dataset storage
├── tests/                         # Unit tests
├── docs/                          # Documentation
├── requirements.txt               # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Technical Features

### Data Processing
- Domain-based data splitting to prevent data leakage
- Multi-format support (CSV, JSON, XLSX, Parquet, COCO annotations)
- Comprehensive feature extraction for each modality
- Strict schema validation and error handling

### Model Quality
- Stratified K-fold cross-validation for all models
- Model calibration using isotonic/Platt scaling
- Automatic feature importance and selection
- Comprehensive performance metrics (Accuracy, F1, Precision, Recall, ROC-AUC)

### Production Features
- Robust error handling and validation
- Memory-efficient deployment
- Complete deployment and maintenance documentation
- Clean, organized, production-ready codebase

## Performance Summary

| Model | Type | Accuracy | Status | Key Features |
|-------|------|----------|---------|--------------|
| Website Fraud | Logistic Regression | 95.2% | Production Ready | URL TF-IDF + Domain Analysis |
| Mobile Risk | Ensemble Fusion | 87.3% | Production Ready | Reviews + PlayStore Metadata |
| APK Malware | Random Forest | 91.7% | Production Ready | API Signatures + Static Analysis |
| Vision Brand | ExtraTrees | 45.5% | Acceptable | COCO Bounding Boxes + Image Metadata |

**Overall System Accuracy**: 79.9% (weighted average)

## Deployment

### Local Development

```bash
# Run the complete system test
python src/cli.py status

# Test individual models
python src/cli.py website-fraud "http://example.com"
```

### Production Deployment

```python
# Load the complete system
from src.models.fraud_detection_system import FraudDetectionSystem
fraud_system = FraudDetectionSystem()

# Get system status
status = fraud_system.get_system_status()
print(f"Models loaded: {status['total_models']}")
print(f"Status: {status['status']}")
```

## API Reference

### FraudDetectionSystem Class

The main class that orchestrates all fraud detection models.

#### Methods

- `predict_website_fraud(url: str) -> Dict`: Predicts fraud probability for a website URL
- `predict_mobile_risk(reviews: str, metadata: Dict) -> Dict`: Assesses mobile app risk
- `predict_apk_malware(features: np.ndarray) -> Dict`: Detects malware in APK files
- `predict_vision_brand(image_features: np.ndarray) -> Dict`: Identifies brand impersonation
- `get_system_status() -> Dict`: Returns system health and model status

## Development

### Adding New Models

1. Implement the model in `src/models/`
2. Add feature extraction in `src/features/`
3. Update the main system in `src/models/fraud_detection_system.py`
4. Add tests in `tests/`
5. Update documentation

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test suite
python -m pytest tests/test_website_fraud.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.

## Team

- **Team Name**: Processing2o
- **University**: NFSU, Goa
- **Members**: 
  - Abhijeet Kumar
  - Haardik Paras Bhagtani
  - Mihir Ranjan

## Support

For technical support or questions, please open an issue on GitHub or refer to the documentation in the `docs/` directory.
