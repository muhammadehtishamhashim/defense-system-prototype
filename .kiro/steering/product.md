# HifazatAI Security System

HifazatAI is a comprehensive AI-powered security monitoring system that integrates multiple AI pipelines into a unified dashboard for real-time threat detection and response.

## Core Purpose
The system processes security data from three specialized AI pipelines:
- **Threat Intelligence**: IOC extraction and risk classification from threat feeds
- **Video Surveillance**: Object detection, tracking, and behavior analysis
- **Border Anomaly Detection**: Trajectory analysis and unusual movement detection

## Key Features
- Real-time alert generation and management
- Multi-pipeline AI processing with configurable confidence thresholds
- Web-based dashboard with live monitoring capabilities
- RESTful API for alert ingestion and system metrics
- Comprehensive evaluation framework for AI model performance
- CPU-optimized for resource-constrained environments (i5 6th gen target)

## Architecture
The system follows a microservices-like architecture with:
- FastAPI backend serving as the central alert broker
- React/TypeScript frontend for monitoring and management
- SQLite database for alert persistence
- Modular pipeline design for easy extension
- Docker containerization for deployment

## Target Environment
Designed for security operations centers requiring:
- Real-time threat monitoring
- Multi-source intelligence correlation
- Performance metrics and system health monitoring
- Scalable alert management with filtering and pagination

## User Personas
- **Security Analysts**: Monitor alerts, investigate threats, manage system configuration
- **SOC Managers**: View system performance, generate reports, oversee operations
- **System Administrators**: Deploy, maintain, and troubleshoot the system

## Alert Types and Priorities
- **Threat Intelligence Alerts**: High/Medium/Low risk classification
- **Video Surveillance Alerts**: Behavior-based (loitering, zone violations, abandoned objects)
- **Border Anomaly Alerts**: Trajectory-based unusual movement detection

## System Capabilities
- **Real-time Processing**: Sub-second alert generation and display
- **Multi-source Integration**: Supports various input formats (JSON, CSV, XML, video streams)
- **Configurable Thresholds**: Adjustable sensitivity for each detection pipeline
- **Historical Analysis**: Alert history with search and filtering capabilities
- **Performance Monitoring**: System health metrics and pipeline status tracking

## Deployment Options
- **Local Development**: Docker Compose for development and testing
- **Production**: Docker containers with resource limits
- **Cloud**: Google Cloud Run, AWS ECS, or Azure Container Instances
- **Edge**: Optimized for resource-constrained edge computing environments

## Success Metrics
- **Detection Accuracy**: >90% precision for threat classification
- **Response Time**: <2 seconds from detection to alert display
- **System Uptime**: >99% availability during operational hours
- **Resource Efficiency**: <6GB RAM usage on target hardware