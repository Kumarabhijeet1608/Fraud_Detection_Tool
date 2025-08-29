# Training Scripts

This directory contains training scripts for all four fraud detection models in the system.

## Overview

The training scripts provide a comprehensive framework for training and evaluating fraud detection models:

1. **Website Fraud Detection** - URL analysis and phishing detection
2. **Mobile Risk Assessment** - App store metadata and review analysis  
3. **APK Malware Detection** - Android application security analysis
4. **Vision Brand Impersonation** - Logo detection and brand verification

## Scripts

### Individual Training Scripts

- **`train_website_fraud.py`** - Trains logistic regression model for website fraud detection
- **`train_mobile_risk.py`** - Trains ensemble models for mobile app risk assessment
- **`train_apk_malware.py`** - Trains random forest model for APK malware detection
- **`train_vision_brand.py`** - Trains ExtraTrees model for vision brand impersonation

### Comprehensive Training Script

- **`train_all_models.py`** - Orchestrates training of all four models with unified interface

## Usage

### Training Individual Models

```bash
# Website fraud detection
python train_website_fraud.py --data ./data/website_data.csv --output ./artifacts/website_fraud_models

# Mobile risk assessment
python train_mobile_risk.py --data ./data/mobile_data.csv --output ./artifacts/mobile_risk_models

# APK malware detection
python train_apk_malware.py --data ./data/apk_data.csv --output ./artifacts/apk_malware_models

# Vision brand impersonation
python train_vision_brand.py --data ./data/vision_data.csv --output ./artifacts/vision_brand_models --coco ./data/annotations.json
```

### Training All Models

```bash
# Train all models at once
python train_all_models.py --model all --data-dir ./data --output-dir ./artifacts

# Train specific model
python train_all_models.py --model website --data ./data/website_data.csv --output ./artifacts/website_fraud_models
```

## Data Requirements

### Website Fraud Detection
- **Format**: CSV, Parquet, or JSON
- **Required Columns**: `url`, `label`
- **Label Values**: Binary (0/1 or string labels)

### Mobile Risk Assessment
- **Format**: CSV, Parquet, or JSON
- **Required Columns**: `reviews`, `label`
- **Optional Columns**: `rating`, `reviews_count`, `downloads`, `price`, `size`, `content_rating`, `category`, `last_updated`, `min_android_version`

### APK Malware Detection
- **Format**: CSV, Parquet, or JSON
- **Required Columns**: Feature columns + `label`
- **Label Values**: Binary (0/1 or string labels)

### Vision Brand Impersonation
- **Format**: CSV, Parquet, or JSON
- **Required Columns**: Feature columns + `label`
- **Optional**: COCO annotation file for enhanced features
- **Label Values**: Binary (0/1 or string labels)

## Output Structure

Each training script creates the following output structure:

```
output_directory/
├── model_name.pkl              # Trained model
├── model_name_scaler.pkl       # Feature scaler
├── model_name_features.json    # Feature names
└── model_name_encoder.pkl      # Label encoder (if applicable)
```

## Model Details

### Website Fraud Detection
- **Algorithm**: Logistic Regression with calibration
- **Features**: URL structure analysis, domain extraction, TF-IDF text processing
- **Preprocessing**: Standard scaling, TF-IDF vectorization
- **Validation**: 5-fold cross-validation with ROC-AUC scoring

### Mobile Risk Assessment
- **Algorithm**: Ensemble fusion (Logistic Regression + Random Forest + Gradient Boosting)
- **Features**: Text sentiment analysis, PlayStore metadata analysis
- **Preprocessing**: TF-IDF for reviews, column transformation for metadata
- **Validation**: 5-fold cross-validation with ROC-AUC scoring

### APK Malware Detection
- **Algorithm**: Random Forest with feature selection
- **Features**: API signatures, permissions, static analysis features
- **Preprocessing**: Standard scaling, statistical feature selection
- **Validation**: 5-fold cross-validation with ROC-AUC scoring

### Vision Brand Impersonation
- **Algorithm**: ExtraTrees with COCO integration
- **Features**: Image features, bounding box analysis, brand text detection
- **Preprocessing**: Standard scaling, statistical feature selection
- **Validation**: 5-fold cross-validation with ROC-AUC scoring

## Dependencies

All training scripts require the following Python packages:

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0
tldextract>=4.0.0
```

## Configuration

### Hyperparameter Tuning
All models use GridSearchCV for hyperparameter optimization with 5-fold cross-validation.

### Feature Selection
APK and Vision models include statistical feature selection using F-test (SelectKBest).

### Model Calibration
Website fraud detection model uses isotonic calibration for better probability estimates.

## Logging

All scripts provide comprehensive logging with:
- Training progress updates
- Model performance metrics
- Feature importance analysis
- Error handling and debugging information

## Error Handling

The training scripts include robust error handling for:
- Missing data files
- Invalid data formats
- Training failures
- Memory issues
- Interrupted training

## Performance Considerations

- **Memory**: Large datasets may require chunked processing
- **Parallelization**: Models use n_jobs=-1 for parallel training where possible
- **Feature Selection**: Reduces memory usage and training time
- **Early Stopping**: Some models support early stopping for faster training

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce feature count or use smaller datasets
2. **Data Format Errors**: Ensure data files match expected format
3. **Missing Dependencies**: Install required packages from requirements.txt
4. **Training Failures**: Check data quality and feature distributions

### Debug Mode

Enable debug logging by modifying the logging level in each script:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Contributing

When adding new training scripts:

1. Follow the established class structure
2. Include comprehensive docstrings
3. Add proper error handling
4. Include logging throughout the pipeline
5. Add the new script to train_all_models.py
6. Update this README with new information

## License

This training framework is part of the Multi-Modal Fraud Detection System.
See the main project LICENSE file for details.
