#!/usr/bin/env python3
"""
Backend initialization script for HifazatAI.
Sets up directories, configuration, and validates the environment.
"""

import os
import sys
from pathlib import Path
from utils.logging import get_logger
from utils.config import ConfigManager
from utils.storage import FileStorageManager
from models.database import create_tables


def create_directory_structure():
    """Create necessary directories for the backend"""
    directories = [
        "config",
        "data",
        "logs", 
        "media/images",
        "media/videos",
        "media/audio",
        "media/documents",
        "media/data",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directory structure")


def initialize_configuration():
    """Initialize configuration files"""
    config_manager = ConfigManager()
    
    # Create sample configuration
    sample_config_path = "config/sample_config.yaml"
    if not Path(sample_config_path).exists():
        config_manager.create_sample_config(sample_config_path)
        print(f"✓ Created sample configuration at {sample_config_path}")
    
    # Save default configuration
    default_config_path = "config/default_config.json"
    if not Path(default_config_path).exists():
        config_manager.save_config(default_config_path)
        print(f"✓ Created default configuration at {default_config_path}")
    
    return config_manager


def initialize_database():
    """Initialize the database"""
    try:
        create_tables()
        print("✓ Database tables created successfully")
        return True
    except Exception as e:
        print(f"✗ Error creating database tables: {str(e)}")
        return False


def initialize_storage():
    """Initialize file storage system"""
    try:
        storage_manager = FileStorageManager()
        stats = storage_manager.get_storage_stats()
        print(f"✓ File storage initialized at {stats['storage_path']}")
        return True
    except Exception as e:
        print(f"✗ Error initializing file storage: {str(e)}")
        return False


def validate_dependencies():
    """Validate that required dependencies are installed"""
    required_packages = [
        "fastapi",
        "uvicorn", 
        "sqlalchemy",
        "pydantic",
        "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"✗ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    else:
        print("✓ All required dependencies are installed")
        return True


def check_environment():
    """Check environment variables and settings"""
    env_vars = {
        "DATABASE_URL": os.getenv("DATABASE_URL", "sqlite:///./hifazat.db"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "API_BASE_URL": os.getenv("API_BASE_URL", "http://localhost:8000")
    }
    
    print("Environment variables:")
    for var, value in env_vars.items():
        print(f"  {var}: {value}")
    
    return True


def run_basic_tests():
    """Run basic functionality tests"""
    try:
        # Test logging
        logger = get_logger("init_test")
        logger.info("Test log message")
        
        # Test configuration
        config_manager = ConfigManager()
        test_value = config_manager.get("pipelines.threat_intelligence.enabled", True)
        
        # Test storage
        storage_manager = FileStorageManager()
        stats = storage_manager.get_storage_stats()
        
        print("✓ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {str(e)}")
        return False


def main():
    """Main initialization function"""
    print("Initializing HifazatAI Backend...")
    print("=" * 50)
    
    success = True
    
    # Step 1: Validate dependencies
    if not validate_dependencies():
        success = False
    
    # Step 2: Create directory structure
    try:
        create_directory_structure()
    except Exception as e:
        print(f"✗ Error creating directories: {str(e)}")
        success = False
    
    # Step 3: Initialize configuration
    try:
        config_manager = initialize_configuration()
    except Exception as e:
        print(f"✗ Error initializing configuration: {str(e)}")
        success = False
    
    # Step 4: Initialize database
    if not initialize_database():
        success = False
    
    # Step 5: Initialize storage
    if not initialize_storage():
        success = False
    
    # Step 6: Check environment
    try:
        check_environment()
    except Exception as e:
        print(f"✗ Error checking environment: {str(e)}")
        success = False
    
    # Step 7: Run basic tests
    if not run_basic_tests():
        success = False
    
    print("=" * 50)
    
    if success:
        print("✓ Backend initialization completed successfully!")
        print("\nNext steps:")
        print("1. Review configuration files in config/")
        print("2. Start the API server: python start_api.py")
        print("3. Run tests: python -m pytest tests/ -v")
        print("4. Check API docs at: http://localhost:8000/docs")
        return 0
    else:
        print("✗ Backend initialization failed!")
        print("Please fix the errors above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())