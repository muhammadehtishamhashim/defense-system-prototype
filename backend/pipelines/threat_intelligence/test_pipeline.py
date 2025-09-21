"""
Unit tests for Threat Intelligence Pipeline components.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from pipelines.threat_intelligence.ioc_extractor import IOCExtractor, IOC
from pipelines.threat_intelligence.classifier import ThreatClassifier
from pipelines.threat_intelligence.pipeline import ThreatIntelligencePipeline
from .sample_data import SAMPLE_JSON_FEED, SAMPLE_CSV_FEED, SAMPLE_XML_FEED, SAMPLE_TEXT_FEED


class TestIOCExtractor:

    def setup_method(self):
        """Set up test fixtures"""
        self.extractor = IOCExtractor()

    def test_extract_from_text_ip(self):
        """Test IP extraction from text"""
        text = "Malicious IP detected: 192.168.1.100"
        iocs = self.extractor.extract_from_text(text, "test")

        assert len(iocs) == 1
        assert iocs[0].type == "ip"
        assert iocs[0].value == "192.168.1.100"

    def test_extract_from_text_domain(self):
        """Test domain extraction from text"""
        text = "Suspicious domain: malicious-site.com"
        iocs = self.extractor.extract_from_text(text, "test")

        assert len(iocs) == 1
        assert iocs[0].type == "domain"
        assert iocs[0].value == "malicious-site.com"

    def test_extract_from_text_cve(self):
        """Test CVE extraction from text"""
        text = "Vulnerability CVE-2023-1234 found"
        iocs = self.extractor.extract_from_text(text, "test")

        assert len(iocs) == 1
        assert iocs[0].type == "cve"
        assert iocs[0].value == "CVE-2023-1234"

    def test_extract_from_text_hash(self):
        """Test hash extraction from text"""
        text = "Malicious file: a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
        iocs = self.extractor.extract_from_text(text, "test")

        # Should extract SHA-256 hash
        sha256_ioc = next((ioc for ioc in iocs if ioc.type == "sha256"), None)
        assert sha256_ioc is not None
        assert sha256_ioc.value == "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"

    def test_parse_json_feed(self):
        """Test JSON feed parsing"""
        iocs = self.extractor.parse_feed(SAMPLE_JSON_FEED, "json", "test")

        assert len(iocs) >= 3  # Should extract at least IP, domain, hash

    def test_parse_csv_feed(self):
        """Test CSV feed parsing"""
        iocs = self.extractor.parse_feed(SAMPLE_CSV_FEED, "csv", "test")

        assert len(iocs) >= 3  # Should extract IP, domain, CVE

    def test_parse_xml_feed(self):
        """Test XML feed parsing"""
        iocs = self.extractor.parse_feed(SAMPLE_XML_FEED, "xml", "test")

        assert len(iocs) >= 2  # Should extract IP and hash

    def test_normalize_iocs(self):
        """Test IOC normalization and deduplication"""
        iocs = [
            IOC(type="ip", value="192.168.1.100", confidence=0.8),
            IOC(type="ip", value="192.168.1.100", confidence=0.9),  # Duplicate
            IOC(type="domain", value="example.com", confidence=0.7)
        ]

        normalized = self.extractor.normalize_iocs(iocs)

        assert len(normalized) == 2  # Should have 2 unique IOCs
        # Should keep the higher confidence
        ip_ioc = next(ioc for ioc in normalized if ioc.type == "ip")
        assert ip_ioc.confidence == 0.9

    def test_ip_validation(self):
        """Test IP address validation"""
        # Valid IP
        assert self.extractor._validate_ip("192.168.1.1") == True

        # Invalid IPs
        assert self.extractor._validate_ip("192.168.1.256") == False
        assert self.extractor._validate_ip("192.168") == False
        assert self.extractor._validate_ip("localhost") == False
        assert self.extractor._validate_ip("127.0.0.1") == False  # Loopback

    def test_domain_validation(self):
        """Test domain validation"""
        assert self.extractor._validate_domain("example.com") == True
        assert self.extractor._validate_domain("sub.example.com") == True
        assert self.extractor._validate_domain("example.") == False
        assert self.extractor._validate_domain("example..com") == False


class TestThreatClassifier:

    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = ThreatClassifier(model_dir="/tmp/test_models")

    def test_classifier_initialization(self):
        """Test classifier initialization"""
        assert self.classifier.risk_levels == ['Low', 'Medium', 'High']
        assert self.classifier.device is not None

    def test_heuristic_risk_label(self):
        """Test heuristic risk labeling"""
        # High risk context
        ioc = IOC(
            type="ip",
            value="192.168.1.1",
            context="Command and control server for malware campaign"
        )
        risk = self.classifier._heuristic_risk_label(ioc)
        assert risk == "High"

        # Medium risk context
        ioc = IOC(
            type="ip",
            value="192.168.1.1",
            context="Suspicious scanning activity detected"
        )
        risk = self.classifier._heuristic_risk_label(ioc)
        assert risk == "Medium"

        # Low risk context
        ioc = IOC(
            type="ip",
            value="192.168.1.1",
            context="Regular network traffic"
        )
        risk = self.classifier._heuristic_risk_label(ioc)
        assert risk == "Low"

    @patch('torch.cuda.is_available')
    def test_extract_bert_features(self, mock_cuda):
        """Test BERT feature extraction"""
        mock_cuda.return_value = False

        texts = ["Test IOC with malware context", "Normal network activity"]
        features = self.classifier.extract_bert_features(texts)

        assert features.shape == (2, 768)  # (samples, features)
        assert features.dtype == 'float32'

    def test_predict_single(self):
        """Test single IOC prediction"""
        ioc = IOC(
            type="ip",
            value="192.168.1.1",
            context="Suspicious activity detected"
        )

        risk_level, confidence = self.classifier.predict_single(ioc)

        assert risk_level in ['Low', 'Medium', 'High']
        assert 0.0 <= confidence <= 1.0


class TestThreatIntelligencePipeline:

    def setup_method(self):
        """Set up test fixtures"""
        self.pipeline = ThreatIntelligencePipeline()

    @patch('pipelines.threat_intelligence.pipeline.ThreatIntelligencePipeline._fetch_feed')
    async def test_fetch_next_feed(self, mock_fetch):
        """Test feed fetching logic"""
        # Mock feed sources
        self.pipeline.feed_sources = [
            {"name": "feed1", "type": "file", "path": "/test/path1"},
            {"name": "feed2", "type": "file", "path": "/test/path2"}
        ]

        mock_fetch.return_value = "test data"

        feed_data = await self.pipeline._fetch_next_feed()

        assert feed_data is not None
        assert feed_data['source'] == "feed1"
        assert feed_data['data'] == "test data"

    def test_create_alert_high_risk(self):
        """Test alert creation for high-risk IOC"""
        processing_result = {
            'source': 'test_feed',
            'iocs': [
                {'type': 'ip', 'value': '192.168.1.100', 'context': 'C2 server'},
                {'type': 'domain', 'value': 'malware.com', 'context': 'Malware distribution'}
            ],
            'classifications': [('High', 0.9), ('Low', 0.3)]
        }

        alert = self.pipeline.create_alert(processing_result)

        assert alert is not None
        assert alert.ioc_type == 'ip'
        assert alert.ioc_value == '192.168.1.100'
        assert alert.risk_level == 'High'
        assert alert.confidence == 0.9

    def test_create_alert_no_high_risk(self):
        """Test alert creation when no high-risk IOCs"""
        processing_result = {
            'source': 'test_feed',
            'iocs': [
                {'type': 'ip', 'value': '192.168.1.100', 'context': 'Normal traffic'}
            ],
            'classifications': [('Low', 0.3)]
        }

        alert = self.pipeline.create_alert(processing_result)

        assert alert is None

    def test_process_input_with_error(self):
        """Test processing input with error"""
        input_data = {
            'source': 'test',
            'data': None
        }

        result = self.pipeline.process_input(input_data)

        assert 'error' in result
        assert result['source'] == 'test'

    @patch('pipelines.threat_intelligence.pipeline.ThreatIntelligencePipeline.ioc_extractor')
    @patch('pipelines.threat_intelligence.pipeline.ThreatIntelligencePipeline.classifier')
    def test_process_input_success(self, mock_classifier, mock_extractor):
        """Test successful input processing"""
        # Mock IOC extraction
        mock_iocs = [
            IOC(type="ip", value="192.168.1.1", context="test"),
            IOC(type="domain", value="example.com", context="test")
        ]
        mock_extractor.parse_feed.return_value = mock_iocs
        mock_extractor.normalize_iocs.return_value = mock_iocs

        # Mock classification
        mock_classifier.predict.return_value = [('High', 0.9), ('Low', 0.3)]

        input_data = {
            'source': 'test_feed',
            'format': 'json',
            'data': '{"test": "data"}'
        }

        result = self.pipeline.process_input(input_data)

        assert 'error' not in result
        assert result['source'] == 'test_feed'
        assert result['total_iocs'] == 2
        assert len(result['classifications']) == 2

    def test_get_pipeline_status(self):
        """Test pipeline status reporting"""
        status = self.pipeline.get_pipeline_status()

        assert 'pipeline_name' in status
        assert 'status' in status
        assert 'metrics' in status
        assert 'threat_specific' in status
        assert status['threat_specific']['feed_sources'] == 0


# Integration test
@pytest.mark.asyncio
async def test_pipeline_integration():
    """Integration test for the complete pipeline"""
    pipeline = ThreatIntelligencePipeline()

    # Test with sample text feed
    input_data = {
        'source': 'sample_text',
        'format': 'text',
        'data': SAMPLE_TEXT_FEED
    }

    result = await pipeline.process_input(input_data)

    # Should extract IOCs from the text
    assert 'total_iocs' in result
    assert result['total_iocs'] > 0

    # Should be able to create alert if high-risk IOCs found
    alert = pipeline.create_alert(result)
    # Alert may or may not be created depending on classification


if __name__ == "__main__":
    pytest.main([__file__])
