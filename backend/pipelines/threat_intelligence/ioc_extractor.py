"""
IOC extraction and parsing module for threat intelligence feeds.
Supports multiple input formats (JSON, CSV, XML) and uses both regex patterns and NLP for IOC extraction.
"""

import re
import json
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from utils.logging import get_pipeline_logger

logger = get_pipeline_logger("threat_intelligence")


@dataclass
class IOC:
    """Indicator of Compromise data structure"""
    type: str  # ip, domain, hash, cve, email, url
    value: str
    confidence: float = 1.0
    source: str = ""
    context: str = ""
    timestamp: Optional[datetime] = None


class IOCExtractor:
    """Extract Indicators of Compromise from various data formats"""

    def __init__(self):
        """Initialize IOC extractor with regex patterns and spaCy model"""
        self.logger = logger

        # Load spaCy model for NER
        try:
            self.nlp: Language = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Regex patterns for IOC extraction
        self.patterns = {
            'ip': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'domain': r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))*)?',
            'md5': r'\b[a-fA-F0-9]{32}\b',
            'sha1': r'\b[a-fA-F0-9]{40}\b',
            'sha256': r'\b[a-fA-F0-9]{64}\b',
            'cve': r'\bCVE-\d{4}-\d{4,}\b'
        }

        # Validation functions for each IOC type
        self.validators = {
            'ip': self._validate_ip,
            'domain': self._validate_domain,
            'email': self._validate_email,
            'url': self._validate_url,
            'md5': self._validate_hash,
            'sha1': self._validate_hash,
            'sha256': self._validate_hash,
            'cve': self._validate_cve
        }

        self.logger.info("IOC extractor initialized")

    def _validate_ip(self, value: str) -> bool:
        """Validate IP address"""
        try:
            parts = value.split('.')
            if len(parts) != 4:
                return False
            for part in parts:
                if not 0 <= int(part) <= 255:
                    return False
            # Exclude private IPs and localhost
            if value.startswith(('10.', '192.168.', '172.', '127.')):
                return False
            return True
        except ValueError:
            return False

    def _validate_domain(self, value: str) -> bool:
        """Validate domain name"""
        if len(value) > 253:
            return False
        if value.endswith('.'):
            return False
        # Basic domain validation
        return re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value) is not None

    def _validate_email(self, value: str) -> bool:
        """Validate email address"""
        return '@' in value and '.' in value.split('@')[1]

    def _validate_url(self, value: str) -> bool:
        """Validate URL"""
        return value.startswith(('http://', 'https://'))

    def _validate_hash(self, value: str) -> bool:
        """Validate file hash"""
        return len(value) in [32, 40, 64] and all(c in '0123456789abcdefABCDEF' for c in value)

    def _validate_cve(self, value: str) -> bool:
        """Validate CVE identifier"""
        return value.startswith('CVE-') and len(value) >= 12

    def extract_from_text(self, text: str, source: str = "") -> List[IOC]:
        """
        Extract IOCs from text using regex patterns and NER

        Args:
            text: Input text to extract IOCs from
            source: Source of the text (feed name, etc.)

        Returns:
            List of extracted IOC objects
        """
        iocs = []
        text_lower = text.lower()

        # Extract using regex patterns
        for ioc_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group()

                # Skip if it's already been extracted as another type
                if self._is_duplicate(iocs, value, ioc_type):
                    continue

                # Validate the IOC
                if self.validators.get(ioc_type, lambda x: False)(value):
                    ioc = IOC(
                        type=ioc_type,
                        value=value,
                        source=source,
                        context=text[max(0, match.start()-50):match.end()+50],
                        timestamp=datetime.now()
                    )
                    iocs.append(ioc)
                    self.logger.debug(f"Extracted {ioc_type}: {value}")

        # Extract using spaCy NER if available
        if self.nlp:
            iocs.extend(self._extract_with_ner(text, source))

        return iocs

    def _extract_with_ner(self, text: str, source: str = "") -> List[IOC]:
        """Extract IOCs using spaCy Named Entity Recognition"""
        if not self.nlp:
            return []

        iocs = []
        doc: Doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'LOC']:
                # Skip common entities that aren't typically IOCs
                continue

            # Convert entity text to potential IOC types
            text_lower = ent.text.lower()

            # Check if it matches any IOC patterns
            for ioc_type, pattern in self.patterns.items():
                if re.search(pattern, ent.text, re.IGNORECASE):
                    if self.validators.get(ioc_type, lambda x: False)(ent.text):
                        ioc = IOC(
                            type=ioc_type,
                            value=ent.text,
                            confidence=0.7,  # Lower confidence for NER extraction
                            source=source,
                            context=text[max(0, ent.start_char-50):ent.end_char+50],
                            timestamp=datetime.now()
                        )
                        iocs.append(ioc)
                        break

        return iocs

    def _is_duplicate(self, existing_iocs: List[IOC], value: str, ioc_type: str) -> bool:
        """Check if IOC is already extracted"""
        for ioc in existing_iocs:
            if ioc.value.lower() == value.lower() and ioc.type == ioc_type:
                return True
        return False

    def parse_feed(self, feed_data: Any, feed_format: str, source: str = "") -> List[IOC]:
        """
        Parse threat intelligence feed in various formats

        Args:
            feed_data: Raw feed data
            feed_format: Format of the feed (json, csv, xml)
            source: Source feed name

        Returns:
            List of extracted IOC objects
        """
        iocs = []

        try:
            if feed_format.lower() == 'json':
                iocs = self._parse_json_feed(feed_data, source)
            elif feed_format.lower() == 'csv':
                iocs = self._parse_csv_feed(feed_data, source)
            elif feed_format.lower() == 'xml':
                iocs = self._parse_xml_feed(feed_data, source)
            else:
                # Treat as plain text
                if isinstance(feed_data, str):
                    iocs = self.extract_from_text(feed_data, source)
                else:
                    self.logger.warning(f"Unsupported feed format: {feed_format}")

        except Exception as e:
            self.logger.error(f"Error parsing {feed_format} feed: {str(e)}")

        return iocs

    def _parse_json_feed(self, feed_data: Any, source: str = "") -> List[IOC]:
        """Parse JSON threat intelligence feed"""
        iocs = []

        try:
            if isinstance(feed_data, str):
                data = json.loads(feed_data)
            else:
                data = feed_data

            # Handle different JSON structures
            if isinstance(data, dict):
                # Single indicator
                iocs.extend(self._extract_iocs_from_dict(data, source))
            elif isinstance(data, list):
                # List of indicators
                for item in data:
                    if isinstance(item, dict):
                        iocs.extend(self._extract_iocs_from_dict(item, source))

            # Extract from all text fields
            if isinstance(data, (dict, list)):
                text_content = json.dumps(data)
                iocs.extend(self.extract_from_text(text_content, source))

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format: {str(e)}")

        return iocs

    def _parse_csv_feed(self, feed_data: str, source: str = "") -> List[IOC]:
        """Parse CSV threat intelligence feed"""
        iocs = []

        try:
            # Try to detect delimiter
            sample = feed_data[:1024]
            delimiter = csv.Sniffer().sniff(sample).delimiter

            reader = csv.DictReader(feed_data.splitlines(), delimiter=delimiter)

            for row in reader:
                iocs.extend(self._extract_iocs_from_dict(row, source))
                # Also extract from raw text
                row_text = ' '.join(str(value) for value in row.values())
                iocs.extend(self.extract_from_text(row_text, source))

        except Exception as e:
            self.logger.error(f"Error parsing CSV: {str(e)}")

        return iocs

    def _parse_xml_feed(self, feed_data: str, source: str = "") -> List[IOC]:
        """Parse XML threat intelligence feed"""
        iocs = []

        try:
            root = ET.fromstring(feed_data)

            # Extract text content from all elements
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    iocs.extend(self.extract_from_text(elem.text, source))

                # Also extract from attributes
                for attr_value in elem.attrib.values():
                    if attr_value.strip():
                        iocs.extend(self.extract_from_text(attr_value, source))

        except ET.ParseError as e:
            self.logger.error(f"Error parsing XML: {str(e)}")

        return iocs

    def _extract_iocs_from_dict(self, data: Dict, source: str = "") -> List[IOC]:
        """Extract IOCs from dictionary data"""
        iocs = []

        # Common field names that might contain IOCs
        ioc_fields = {
            'ip': ['ip', 'ip_address', 'src_ip', 'dst_ip'],
            'domain': ['domain', 'hostname', 'url'],
            'hash': ['hash', 'md5', 'sha1', 'sha256'],
            'cve': ['cve', 'vulnerability']
        }

        for field_value in data.values():
            if isinstance(field_value, str) and field_value.strip():
                iocs.extend(self.extract_from_text(field_value, source))

        return iocs

    def normalize_iocs(self, iocs: List[IOC]) -> List[IOC]:
        """
        Normalize and deduplicate IOCs

        Args:
            iocs: List of IOC objects

        Returns:
            Normalized and deduplicated list of IOCs
        """
        normalized = {}
        normalized_iocs = []

        for ioc in iocs:
            # Create normalization key
            key = f"{ioc.type}:{ioc.value.lower().strip()}"

            if key in normalized:
                # Update existing IOC with higher confidence
                existing = normalized[key]
                if ioc.confidence > existing.confidence:
                    existing.confidence = ioc.confidence
                    existing.context = ioc.context
                    existing.source = ioc.source
            else:
                # Add new normalized IOC
                normalized_ioc = IOC(
                    type=ioc.type,
                    value=ioc.value.strip(),
                    confidence=ioc.confidence,
                    source=ioc.source,
                    context=ioc.context,
                    timestamp=ioc.timestamp
                )
                normalized[key] = normalized_ioc
                normalized_iocs.append(normalized_ioc)

        return normalized_iocs
