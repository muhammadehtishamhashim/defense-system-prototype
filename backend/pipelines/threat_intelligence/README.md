# Threat Intelligence Pipeline

The Threat Intelligence Pipeline is a core component of the HifazatAI system that processes threat intelligence feeds, extracts Indicators of Compromise (IOCs), classifies threats by risk level, and generates actionable alerts for the security team.

## Features

- **Multi-format Feed Processing**: Supports JSON, CSV, XML, and plain text threat intelligence feeds
- **Advanced IOC Extraction**: Uses both regex patterns and spaCy Named Entity Recognition for comprehensive IOC detection
- **Intelligent Threat Classification**: Combines DistilBERT embeddings with TF-IDF features and Random Forest classification
- **Risk-based Alerting**: Generates alerts only for high-risk IOCs with configurable thresholds
- **Scalable Architecture**: Built on the base pipeline framework with async processing support
- **Comprehensive Validation**: Validates and normalizes IOCs to prevent duplicates and false positives

## Architecture

### Components

1. **IOC Extractor** (`ioc_extractor.py`)
   - Parses threat intelligence feeds in multiple formats
   - Extracts IOCs using regex patterns and NLP
   - Validates and normalizes extracted IOCs

2. **Threat Classifier** (`classifier.py`)
   - Uses DistilBERT for contextual embeddings
   - Combines with TF-IDF features for classification
   - Classifies IOCs into High, Medium, Low risk levels

3. **Threat Intelligence Pipeline** (`pipeline.py`)
   - Main pipeline orchestrator extending BasePipeline
   - Manages feed ingestion and processing workflow
   - Generates alerts for high-risk IOCs

### IOC Types Supported

- **IP Addresses**: IPv4 addresses (excluding private/localhost ranges)
- **Domain Names**: Fully qualified domain names with validation
- **Email Addresses**: Standard email format validation
- **URLs**: HTTP/HTTPS URLs with proper formatting
- **File Hashes**: MD5, SHA-1, SHA-256 hash validation
- **CVE Identifiers**: Standard CVE format (CVE-YYYY-NNNN+)

## Installation

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install spaCy model for NER
python -m spacy download en_core_web_sm

# Install PyTorch (for DistilBERT)
pip install torch torchvision torchaudio
```

### Required Dependencies

```txt
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
spacy==3.7.2
transformers==4.35.2
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.3
aiohttp==3.9.1
python-multipart==0.0.6
```

## Configuration

Create a configuration file `config/threat_intelligence_config.json`:

```json
{
  "threat_intelligence": {
    "confidence_threshold": 0.7,
    "processing_interval": 5.0,
    "risk_threshold": 0.8,
    "feed_check_interval": 300,
    "batch_size": 100,
    "feed_sources": [
      {
        "name": "threat_feed_1",
        "type": "url",
        "url": "https://example.com/threat-feed.json",
        "format": "json",
        "enabled": true
      }
    ],
    "model_config": {
      "model_dir": "models/threat_classifier",
      "load_pretrained": true,
      "train_on_startup": false
    },
    "api_base_url": "http://localhost:8000",
    "enable_async_processing": true,
    "max_concurrent_feeds": 5
  }
}
```

## Usage

### Basic Usage

```python
from pipelines.threat_intelligence import ThreatIntelligencePipeline

# Create pipeline instance
pipeline = ThreatIntelligencePipeline()

# Process a threat feed
input_data = {
    'source': 'my_threat_feed',
    'format': 'json',
    'data': '{"indicators": [...]}'
}

result = await pipeline.process_input(input_data)

# Check for alerts
alert = pipeline.create_alert(result)
if alert:
    print(f"High-risk alert: {alert.ioc_value}")
```

### Running as a Service

```python
import asyncio
from pipelines.threat_intelligence import ThreatIntelligencePipeline

async def main():
    pipeline = ThreatIntelligencePipeline()
    await pipeline.start()

if __name__ == "__main__":
    asyncio.run(main())
```

### Training the Classifier

```python
from pipelines.threat_intelligence import ThreatIntelligencePipeline, IOC

# Create sample IOCs for training
sample_iocs = [
    IOC(type="ip", value="192.168.1.100", context="C2 server"),
    IOC(type="domain", value="malware.com", context="Malware distribution")
]

pipeline = ThreatIntelligencePipeline()
metrics = pipeline.classifier.train(sample_iocs, test_size=0.2)

print(f"Training accuracy: {metrics['accuracy']}")
```

## API Reference

### IOCExtractor

```python
extractor = IOCExtractor()

# Extract IOCs from text
iocs = extractor.extract_from_text("Malicious IP: 192.168.1.100", "source")

# Parse structured feed
iocs = extractor.parse_feed(feed_data, "json", "source")

# Normalize and deduplicate
normalized_iocs = extractor.normalize_iocs(iocs)
```

### ThreatClassifier

```python
classifier = ThreatClassifier()

# Predict risk level
risk_level, confidence = classifier.predict_single(ioc)

# Train with labeled data
metrics = classifier.train(iocs, test_size=0.2)

# Save/load model
classifier.save_model()
classifier.load_model()
```

### ThreatIntelligencePipeline

```python
pipeline = ThreatIntelligencePipeline(config_path="config.json")

# Get pipeline status
status = pipeline.get_pipeline_status()

# Process feed data
result = await pipeline.process_input(input_data)

# Create alert from results
alert = pipeline.create_alert(result)
```

## Feed Format Examples

### JSON Format

```json
{
  "indicators": [
    {
      "type": "ip",
      "value": "192.168.1.100",
      "context": "Command and control server",
      "confidence": 0.95
    }
  ],
  "timestamp": "2024-01-15T10:30:00Z",
  "source": "Threat Intelligence Feed"
}
```

### CSV Format

```csv
indicator_type,indicator_value,description,confidence
ip,192.168.1.100,C2 server detected,0.95
domain,malware-site.com,Malware distribution,0.85
```

### XML Format

```xml
<threat_feed>
    <indicator>
        <type>ip</type>
        <value>192.168.1.100</value>
        <description>C2 server</description>
        <confidence>0.95</confidence>
    </indicator>
</threat_feed>
```

## Testing

Run the unit tests:

```bash
# Run all tests
python -m pytest pipelines/threat_intelligence/test_pipeline.py

# Run specific test class
python -m pytest pipelines/threat_intelligence/test_pipeline.py::TestIOCExtractor

# Run with coverage
python -m pytest pipelines/threat_intelligence/test_pipeline.py --cov=.
```

## Demo

Run the demo script to see the pipeline in action:

```bash
# Basic demo
python pipelines/threat_intelligence/demo.py

# Include classifier training
python pipelines/threat_intelligence/demo.py --train
```

## Performance Considerations

- **Memory Usage**: DistilBERT model requires ~250MB RAM
- **Processing Speed**: ~100 IOCs/second on CPU, ~500 IOCs/second on GPU
- **Feed Polling**: Configurable interval (default 5 minutes)
- **Batch Processing**: Processes IOCs in configurable batches

## Error Handling

The pipeline includes comprehensive error handling:

- **Feed Parsing Errors**: Logs and continues processing
- **Model Loading Failures**: Falls back to heuristic classification
- **API Communication Errors**: Retries with exponential backoff
- **Resource Exhaustion**: Graceful degradation with reduced functionality

## Security Considerations

- **Input Validation**: All inputs are validated and sanitized
- **IOC Validation**: Prevents malicious IOC injection
- **Resource Limits**: Configurable limits on processing resources
- **Audit Logging**: All operations are logged for security review

## Monitoring and Logging

The pipeline provides detailed metrics and logging:

- **Processing Metrics**: Items processed, success rates, processing times
- **Classification Accuracy**: Model performance and confidence scores
- **Alert Generation**: Number of alerts generated and their risk levels
- **Error Tracking**: Detailed error logging with context

## Troubleshooting

### Common Issues

1. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **CUDA Not Available**
   - The pipeline automatically falls back to CPU processing
   - Performance will be slower but functionality is preserved

3. **Feed Parsing Errors**
   - Check feed format matches configuration
   - Verify feed data is valid JSON/CSV/XML
   - Check network connectivity for URL feeds

4. **Low Classification Accuracy**
   - Train the model with more labeled data
   - Adjust confidence thresholds
   - Review feature extraction settings

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger("threat_intelligence").setLevel(logging.DEBUG)
```

## Contributing

1. Follow the existing code structure and naming conventions
2. Add unit tests for new functionality
3. Update documentation for API changes
4. Ensure all tests pass before submitting

## License

This component is part of the HifazatAI system and follows the same licensing terms.
