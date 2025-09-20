"""
Configuration management utilities for HifazatAI.
Handles loading, validation, and management of configuration files.
"""

import json
import yaml
import os
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict
from utils.logging import get_logger


@dataclass
class PipelineConfig:
    """Base configuration for pipelines"""
    enabled: bool = True
    confidence_threshold: float = 0.5
    processing_interval: float = 1.0
    api_base_url: str = "http://localhost:8000"
    log_level: str = "INFO"


@dataclass
class ThreatIntelligenceConfig(PipelineConfig):
    """Configuration for threat intelligence pipeline"""
    feed_sources: list = None
    risk_thresholds: dict = None
    model_path: str = "models/threat_classifier"
    ioc_extraction_patterns: dict = None
    
    def __post_init__(self):
        if self.feed_sources is None:
            self.feed_sources = []
        if self.risk_thresholds is None:
            self.risk_thresholds = {"high": 0.8, "medium": 0.5, "low": 0.2}
        if self.ioc_extraction_patterns is None:
            self.ioc_extraction_patterns = {
                "ip": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
                "domain": r"\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*\b",
                "hash_md5": r"\b[a-fA-F0-9]{32}\b",
                "hash_sha1": r"\b[a-fA-F0-9]{40}\b",
                "hash_sha256": r"\b[a-fA-F0-9]{64}\b"
            }


@dataclass
class VideoSurveillanceConfig(PipelineConfig):
    """Configuration for video surveillance pipeline"""
    input_sources: list = None
    detection_classes: list = None
    tracking_enabled: bool = True
    behavior_rules: dict = None
    model_path: str = "models/yolov8n.pt"
    snapshot_storage_path: str = "media/snapshots"
    
    def __post_init__(self):
        if self.input_sources is None:
            self.input_sources = []
        if self.detection_classes is None:
            self.detection_classes = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]
        if self.behavior_rules is None:
            self.behavior_rules = {
                "loitering": {"threshold_seconds": 30, "enabled": True},
                "zone_violation": {"zones": [], "enabled": True},
                "abandoned_object": {"threshold_seconds": 60, "enabled": True}
            }


@dataclass
class BorderAnomalyConfig(PipelineConfig):
    """Configuration for border anomaly detection pipeline"""
    sensitivity: float = 0.7
    min_trajectory_length: int = 10
    feature_weights: dict = None
    model_path: str = "models/anomaly_detector"
    normal_patterns_path: str = "data/normal_patterns.json"
    
    def __post_init__(self):
        if self.feature_weights is None:
            self.feature_weights = {
                "speed": 0.3,
                "curvature": 0.2,
                "direction_change": 0.2,
                "duration": 0.15,
                "entry_angle": 0.15
            }


class ConfigManager:
    """Configuration manager for loading and managing pipeline configurations"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        self.logger = get_logger(__name__)
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}
        
        # Default configuration directory
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
        else:
            self._load_default_configs()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration dictionary
        """
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                self.logger.warning(f"Configuration file not found: {config_path}")
                return {}
            
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
            
            self.config_data.update(config_data)
            self.logger.info(f"Loaded configuration from {config_path}")
            return config_data
            
        except Exception as e:
            self.logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            return {}
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Optional path to save configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            save_path = config_path or self.config_path or "config/hifazat_config.json"
            config_file = Path(save_path)
            
            # Create directory if it doesn't exist
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                if config_file.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config_data, f, indent=2, default=str)
            
            self.logger.info(f"Saved configuration to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config_data
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        self.logger.debug(f"Set configuration {key} = {value}")
    
    def update(self, updates: Dict[str, Any]):
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def get_pipeline_config(self, pipeline_name: str) -> Union[PipelineConfig, Dict[str, Any]]:
        """
        Get configuration for a specific pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            Pipeline configuration object or dictionary
        """
        config_data = self.get(f"pipelines.{pipeline_name}", {})
        
        # Create appropriate configuration object
        if pipeline_name == "threat_intelligence":
            return ThreatIntelligenceConfig(**config_data)
        elif pipeline_name == "video_surveillance":
            return VideoSurveillanceConfig(**config_data)
        elif pipeline_name == "border_anomaly":
            return BorderAnomalyConfig(**config_data)
        else:
            return PipelineConfig(**config_data)
    
    def set_pipeline_config(self, pipeline_name: str, config: Union[PipelineConfig, Dict[str, Any]]):
        """
        Set configuration for a specific pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            config: Configuration object or dictionary
        """
        if isinstance(config, PipelineConfig):
            config_dict = asdict(config)
        else:
            config_dict = config
        
        self.set(f"pipelines.{pipeline_name}", config_dict)
    
    def _load_default_configs(self):
        """Load default configurations for all pipelines"""
        default_config = {
            "pipelines": {
                "threat_intelligence": asdict(ThreatIntelligenceConfig()),
                "video_surveillance": asdict(VideoSurveillanceConfig()),
                "border_anomaly": asdict(BorderAnomalyConfig())
            },
            "api": {
                "base_url": "http://localhost:8000",
                "timeout": 30
            },
            "storage": {
                "media_path": "media",
                "data_path": "data",
                "models_path": "models"
            },
            "logging": {
                "level": "INFO",
                "file": "logs/hifazat.log"
            }
        }
        
        self.config_data = default_config
        self.logger.info("Loaded default configuration")
    
    def validate_config(self) -> Dict[str, List[str]]:
        """
        Validate current configuration.
        
        Returns:
            Dictionary of validation errors by section
        """
        errors = {}
        
        # Validate pipeline configurations
        for pipeline_name in ["threat_intelligence", "video_surveillance", "border_anomaly"]:
            pipeline_errors = []
            config = self.get(f"pipelines.{pipeline_name}", {})
            
            # Check required fields
            if "confidence_threshold" in config:
                threshold = config["confidence_threshold"]
                if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                    pipeline_errors.append("confidence_threshold must be a number between 0 and 1")
            
            if "processing_interval" in config:
                interval = config["processing_interval"]
                if not isinstance(interval, (int, float)) or interval <= 0:
                    pipeline_errors.append("processing_interval must be a positive number")
            
            if pipeline_errors:
                errors[pipeline_name] = pipeline_errors
        
        # Validate API configuration
        api_config = self.get("api", {})
        api_errors = []
        
        if "base_url" in api_config and not isinstance(api_config["base_url"], str):
            api_errors.append("base_url must be a string")
        
        if api_errors:
            errors["api"] = api_errors
        
        return errors
    
    def create_sample_config(self, output_path: str = "config/sample_config.yaml"):
        """
        Create a sample configuration file with all available options.
        
        Args:
            output_path: Path to save the sample configuration
        """
        sample_config = {
            "pipelines": {
                "threat_intelligence": {
                    "enabled": True,
                    "confidence_threshold": 0.7,
                    "processing_interval": 5.0,
                    "feed_sources": [
                        "https://example.com/threat_feed.json",
                        "/path/to/local/feed.csv"
                    ],
                    "risk_thresholds": {
                        "high": 0.8,
                        "medium": 0.5,
                        "low": 0.2
                    },
                    "model_path": "models/threat_classifier"
                },
                "video_surveillance": {
                    "enabled": True,
                    "confidence_threshold": 0.6,
                    "processing_interval": 0.1,
                    "input_sources": [
                        "/path/to/video.mp4",
                        "rtsp://camera1.local/stream"
                    ],
                    "detection_classes": ["person", "car", "truck"],
                    "behavior_rules": {
                        "loitering": {"threshold_seconds": 30, "enabled": True},
                        "zone_violation": {"zones": [], "enabled": True}
                    },
                    "model_path": "models/yolov8n.pt"
                },
                "border_anomaly": {
                    "enabled": True,
                    "confidence_threshold": 0.5,
                    "processing_interval": 1.0,
                    "sensitivity": 0.7,
                    "min_trajectory_length": 10,
                    "feature_weights": {
                        "speed": 0.3,
                        "curvature": 0.2,
                        "direction_change": 0.2
                    }
                }
            },
            "api": {
                "base_url": "http://localhost:8000",
                "timeout": 30
            },
            "storage": {
                "media_path": "media",
                "data_path": "data",
                "models_path": "models"
            },
            "logging": {
                "level": "INFO",
                "file": "logs/hifazat.log"
            }
        }
        
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                yaml.dump(sample_config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Created sample configuration at {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating sample configuration: {str(e)}")
            return False


# Global configuration manager instance
config_manager = ConfigManager()