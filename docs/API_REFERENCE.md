# API Reference

## FraudDetectionSystem Class

The main class that orchestrates all fraud detection models.

### Constructor

```python
FraudDetectionSystem()
```

Initializes the fraud detection system and loads all four models.

### Methods

#### predict_website_fraud(url: str) -> Dict[str, Any]

Predicts fraud probability for a website URL.

**Parameters:**
- `url` (str): The URL to analyze

**Returns:**
- Dictionary containing:
  - `url`: The analyzed URL
  - `prediction`: 'fraudulent' or 'legitimate'
  - `fraud_probability`: Probability of fraud (0.0 to 1.0)
  - `confidence`: Confidence in the prediction
  - `model_info`: Information about the model used

**Example:**
```python
result = fraud_system.predict_website_fraud("http://example.com")
print(f"Prediction: {result['prediction']}")
print(f"Fraud Probability: {result['fraud_probability']:.2%}")
```

#### predict_mobile_risk(reviews: str, metadata: Dict[str, Any]) -> Dict[str, Any]

Assesses mobile app risk based on reviews and metadata.

**Parameters:**
- `reviews` (str): Text reviews for the app
- `metadata` (Dict): PlayStore metadata including rating, downloads, price, size

**Returns:**
- Dictionary containing:
  - `risk_level`: 'low', 'medium', or 'high'
  - `risk_probability`: Probability of risk (0.0 to 1.0)
  - `confidence`: Confidence in the assessment
  - `reviews_sentiment`: Sentiment analysis score
  - `metadata_risk`: Risk score from metadata
  - `model_info`: Information about the model used

**Example:**
```python
metadata = {
    'rating': 4.5,
    'downloads': 1000,
    'price': 0.0,
    'size': 10.0
}
result = fraud_system.predict_mobile_risk("Great app!", metadata)
print(f"Risk Level: {result['risk_level']}")
```

#### predict_apk_malware(features: np.ndarray) -> Dict[str, Any]

Detects malware in APK files based on extracted features.

**Parameters:**
- `features` (np.ndarray): Feature vector for the APK

**Returns:**
- Dictionary containing:
  - `prediction`: 'malicious' or 'benign'
  - `malware_probability`: Probability of malware (0.0 to 1.0)
  - `confidence`: Confidence in the detection
  - `model_info`: Information about the model used

**Example:**
```python
features = np.array([0.1, 0.2, 0.3, ...])  # APK features
result = fraud_system.predict_apk_malware(features)
print(f"Prediction: {result['prediction']}")
```

#### predict_vision_brand(image_features: np.ndarray) -> Dict[str, Any]

Identifies brand impersonation in visual content.

**Parameters:**
- `image_features` (np.ndarray): Feature vector for the image

**Returns:**
- Dictionary containing:
  - `prediction`: 'impersonation' or 'legitimate'
  - `impersonation_probability`: Probability of impersonation (0.0 to 1.0)
  - `confidence`: Confidence in the detection
  - `model_info`: Information about the model used

**Example:**
```python
features = np.array([0.1, 0.2, 0.3, ...])  # Image features
result = fraud_system.predict_vision_brand(features)
print(f"Prediction: {result['prediction']}")
```

#### get_system_status() -> Dict[str, Any]

Gets the current status of the fraud detection system.

**Returns:**
- Dictionary containing:
  - `status`: System status ('operational' or 'not_initialized')
  - `total_models`: Number of loaded models
  - `models`: List of model names
  - `model_info`: Information about all models

**Example:**
```python
status = fraud_system.get_system_status()
print(f"System Status: {status['status']}")
print(f"Models Loaded: {status['total_models']}")
```

#### get_model_info(model_name: str) -> Optional[Dict[str, str]]

Gets information about a specific model.

**Parameters:**
- `model_name` (str): Name of the model

**Returns:**
- Model information dictionary or None if not found

**Example:**
```python
info = fraud_system.get_model_info('website_fraud')
if info:
    print(f"Model Type: {info['type']}")
    print(f"Accuracy: {info['accuracy']}")
```

## Error Handling

All prediction methods return a dictionary with an `error` key if something goes wrong:

```python
result = fraud_system.predict_website_fraud("invalid_url")
if 'error' in result:
    print(f"Error: {result['error']}")
else:
    # Process successful result
    print(f"Prediction: {result['prediction']}")
```

## Model Information

Each model provides the following information:

- `type`: Algorithm type (e.g., "Logistic Regression", "Random Forest")
- `accuracy`: Validation accuracy (e.g., "95.2%")
- `features`: Description of features used
- `status`: Model status (e.g., "Production Ready", "Acceptable")

## Usage Examples

### Basic Usage

```python
from src.models.fraud_detection_system import FraudDetectionSystem

# Initialize the system
fraud_system = FraudDetectionSystem()

# Check system status
status = fraud_system.get_system_status()
print(f"System Status: {status['status']}")

# Website fraud detection
url_result = fraud_system.predict_website_fraud("http://example.com")
print(f"URL Analysis: {url_result['prediction']}")

# Mobile risk assessment
metadata = {'rating': 4.0, 'downloads': 500}
mobile_result = fraud_system.predict_mobile_risk("Good app", metadata)
print(f"Risk Level: {mobile_result['risk_level']}")
```

### Batch Processing

```python
# Process multiple URLs
urls = ["http://example1.com", "http://example2.com", "http://example3.com"]
results = []

for url in urls:
    result = fraud_system.predict_website_fraud(url)
    results.append(result)

# Analyze results
fraudulent_count = sum(1 for r in results if r.get('prediction') == 'fraudulent')
print(f"Found {fraudulent_count} fraudulent URLs out of {len(urls)}")
```

### Error Handling

```python
try:
    result = fraud_system.predict_website_fraud("invalid_url")
    if 'error' in result:
        print(f"Analysis failed: {result['error']}")
    else:
        print(f"Success: {result['prediction']}")
except Exception as e:
    print(f"Unexpected error: {e}")
```
