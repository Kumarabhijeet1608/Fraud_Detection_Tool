# Deployment Guide

This guide covers deploying the Multi-Modal Fraud Detection System in various environments.

## Prerequisites

- Python 3.8 or higher
- Access to the trained model artifacts
- Sufficient disk space for model files
- Memory requirements: Minimum 4GB RAM, recommended 8GB+

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Kumarabhijeet1608/Fraud_Detection_Tool.git
cd Fraud_Detection_Tool
```

### 2. Create Virtual Environment

```bash
# Using venv (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### 4. Verify Installation

```bash
# Test the CLI
python src/cli.py status

# Test individual models
python src/cli.py website-fraud "http://example.com"
```

## Production Deployment

### 1. Server Requirements

- **Operating System**: Linux (Ubuntu 18.04+ recommended) or Windows Server
- **Python**: 3.8+
- **Memory**: 8GB+ RAM
- **Storage**: 10GB+ free space
- **Network**: Stable internet connection for model updates

### 2. Production Installation

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv

# Create application directory
sudo mkdir -p /opt/fraud-detection
sudo chown $USER:$USER /opt/fraud-detection
cd /opt/fraud-detection

# Clone repository
git clone https://github.com/Kumarabhijeet1608/Fraud_Detection_Tool.git .
git checkout main

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
# Production settings
ENVIRONMENT=production
LOG_LEVEL=INFO
MODEL_CACHE_SIZE=1000
API_RATE_LIMIT=1000

# Model paths (adjust as needed)
WEBSITE_MODEL_PATH=artifacts/website_fraud_models/
MOBILE_MODEL_PATH=artifacts/mobile_risk_models/
APK_MODEL_PATH=artifacts/apk_malware_models/
VISION_MODEL_PATH=artifacts/vision_brand_models/
```

### 4. Systemd Service (Linux)

Create `/etc/systemd/system/fraud-detection.service`:

```ini
[Unit]
Description=Fraud Detection System
After=network.target

[Service]
Type=simple
User=fraud-detection
WorkingDirectory=/opt/fraud-detection
Environment=PATH=/opt/fraud-detection/venv/bin
ExecStart=/opt/fraud-detection/venv/bin/python src/cli.py status
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable fraud-detection
sudo systemctl start fraud-detection
sudo systemctl status fraud-detection
```

## Docker Deployment

### 1. Dockerfile

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Create non-root user
RUN useradd -m -u 1000 fraud-detection
USER fraud-detection

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from src.models.fraud_detection_system import FraudDetectionSystem; FraudDetectionSystem()" || exit 1

# Default command
CMD ["python", "src/cli.py", "status"]
```

### 2. Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  fraud-detection:
    build: .
    volumes:
      - ./artifacts:/app/artifacts
      - ./logs:/app/logs
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "from src.models.fraud_detection_system import FraudDetectionSystem; FraudDetectionSystem()"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 3. Build and Run

```bash
# Build the image
docker-compose build

# Run the services
docker-compose up -d

# Check logs
docker-compose logs -f fraud-detection
```

## Monitoring and Logging

### 1. Logging Configuration

The system uses Python's built-in logging. Configure logging in production:

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/fraud_detection.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
```

### 2. Health Checks

Implement health check endpoints:

```python
def health_check():
    """Health check endpoint for load balancers."""
    try:
        fraud_system = FraudDetectionSystem()
        status = fraud_system.get_system_status()
        
        if status['status'] == 'operational':
            return {'status': 'healthy', 'models_loaded': status['total_models']}
        else:
            return {'status': 'unhealthy', 'error': 'Models not loaded'}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}
```

### 3. Metrics Collection

Collect performance metrics:

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Log execution time
        execution_time = end_time - start_time
        logging.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper
```

## Security Considerations

### 1. Access Control

- Restrict access to model artifacts
- Use environment variables for sensitive configuration
- Implement API rate limiting
- Use HTTPS in production

### 2. Model Security

- Validate input data before processing
- Implement input sanitization
- Monitor for adversarial attacks
- Regular model updates and security patches

### 3. Network Security

- Use firewalls to restrict access
- Implement network segmentation
- Monitor network traffic for anomalies
- Regular security audits

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check file permissions
   - Verify model file paths
   - Ensure sufficient disk space

2. **Memory Issues**
   - Monitor memory usage
   - Implement model unloading for unused models
   - Use model quantization if needed

3. **Performance Issues**
   - Profile code execution
   - Implement caching strategies
   - Use async processing where appropriate

### Debug Mode

Enable debug mode for troubleshooting:

```bash
export LOG_LEVEL=DEBUG
export ENVIRONMENT=development
python src/cli.py status
```

### Support

For deployment issues:
1. Check the logs for error messages
2. Verify system requirements
3. Test with minimal configuration
4. Open an issue on GitHub with detailed error information
