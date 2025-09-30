"""
Mock Alert Generator
Generates realistic video alerts every 30 seconds for demonstration purposes.
"""

import asyncio
import uuid
import random
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from models.alerts import VideoAlert, AlertStatus
from utils.logging import get_logger

logger = get_logger(__name__)


class AlertScenario(str, Enum):
    """Types of mock alert scenarios"""
    LOITERING = "loitering"
    ZONE_VIOLATION = "zone_violation"
    ABANDONED_OBJECT = "abandoned_object"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    CROWD_DETECTION = "crowd_detection"


class MockAlertGenerator:
    """Generates mock video alerts for demonstration purposes"""
    
    def __init__(self):
        """Initialize mock alert generator"""
        self.active_generators: Dict[str, Dict[str, Any]] = {}
        self.alert_scenarios = list(AlertScenario)
        self.alert_callback = None
        
        # Mock data for realistic alerts
        self.locations = [
            "Main Entrance", "Parking Lot", "Perimeter Fence", 
            "Loading Dock", "Emergency Exit", "Reception Area"
        ]
        
        self.event_descriptions = {
            AlertScenario.LOITERING: [
                "Person detected loitering near entrance for extended period",
                "Individual remaining stationary in restricted area",
                "Suspicious loitering behavior detected in parking area"
            ],
            AlertScenario.ZONE_VIOLATION: [
                "Unauthorized access to restricted zone detected",
                "Person entered secure area without authorization",
                "Zone boundary violation in sensitive area"
            ],
            AlertScenario.ABANDONED_OBJECT: [
                "Unattended bag detected in public area",
                "Suspicious object left unattended",
                "Abandoned package detected near entrance"
            ],
            AlertScenario.SUSPICIOUS_BEHAVIOR: [
                "Unusual movement pattern detected",
                "Person exhibiting suspicious behavior",
                "Abnormal activity detected in monitored area"
            ],
            AlertScenario.CROWD_DETECTION: [
                "Large crowd gathering detected",
                "Unusual crowd density in area",
                "Multiple people congregating in restricted zone"
            ]
        }
        
        logger.info("Mock alert generator initialized")
    
    def set_alert_callback(self, callback):
        """
        Set callback function for generated alerts
        
        Args:
            callback: Function to call when alert is generated
        """
        self.alert_callback = callback
        logger.info("Alert callback set for mock generator")
    
    async def start_mock_alerts(self, session_id: str, interval: int = 30) -> bool:
        """
        Start generating mock alerts for a session
        
        Args:
            session_id: Analysis session ID
            interval: Interval between alerts in seconds (default: 30)
            
        Returns:
            True if started successfully, False otherwise
        """
        if session_id in self.active_generators:
            logger.warning(f"Mock alert generator already running for session {session_id}")
            return False
        
        try:
            # Create generator data
            generator_data = {
                'session_id': session_id,
                'interval': interval,
                'alerts_generated': 0,
                'started_at': datetime.now(),
                'stop_event': asyncio.Event(),
                'task': None
            }
            
            self.active_generators[session_id] = generator_data
            
            # Start generation task
            generator_data['task'] = asyncio.create_task(
                self._generate_alerts_loop(session_id)
            )
            
            logger.info(f"Started mock alert generation for session {session_id} (interval: {interval}s)")
            return True
            
        except Exception as e:
            logger.error(f"Error starting mock alert generation for session {session_id}: {str(e)}")
            if session_id in self.active_generators:
                del self.active_generators[session_id]
            return False
    
    async def stop_mock_alerts(self, session_id: str) -> bool:
        """
        Stop generating mock alerts for a session
        
        Args:
            session_id: Analysis session ID
            
        Returns:
            True if stopped successfully, False if session not found
        """
        if session_id not in self.active_generators:
            logger.warning(f"Mock alert generator not found for session {session_id}")
            return False
        
        try:
            generator_data = self.active_generators[session_id]
            
            # Signal stop
            generator_data['stop_event'].set()
            
            # Cancel task if running
            if generator_data['task'] and not generator_data['task'].done():
                generator_data['task'].cancel()
                try:
                    await generator_data['task']
                except asyncio.CancelledError:
                    pass
            
            # Remove from active generators
            del self.active_generators[session_id]
            
            logger.info(f"Stopped mock alert generation for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping mock alert generation for session {session_id}: {str(e)}")
            return False
    
    def generate_realistic_alert(self, session_id: str) -> VideoAlert:
        """
        Generate a realistic mock video alert
        
        Args:
            session_id: Analysis session ID
            
        Returns:
            VideoAlert object with realistic mock data
        """
        # Select random scenario
        scenario = random.choice(self.alert_scenarios)
        
        # Generate realistic bounding box coordinates
        bbox_x = random.randint(50, 400)
        bbox_y = random.randint(50, 300)
        bbox_width = random.randint(80, 200)
        bbox_height = random.randint(100, 250)
        
        # Generate confidence score (higher for more "obvious" detections)
        confidence_ranges = {
            AlertScenario.ZONE_VIOLATION: (0.85, 0.95),
            AlertScenario.ABANDONED_OBJECT: (0.75, 0.90),
            AlertScenario.LOITERING: (0.70, 0.85),
            AlertScenario.SUSPICIOUS_BEHAVIOR: (0.60, 0.80),
            AlertScenario.CROWD_DETECTION: (0.80, 0.95)
        }
        
        min_conf, max_conf = confidence_ranges.get(scenario, (0.60, 0.90))
        confidence = round(random.uniform(min_conf, max_conf), 2)
        
        # Generate track ID
        track_id = random.randint(1, 50)
        
        # Select random location and description
        location = random.choice(self.locations)
        description = random.choice(self.event_descriptions[scenario])
        
        # Create a mock snapshot
        snapshot_filename = f"mock_{uuid.uuid4().hex[:8]}.jpg"
        snapshot_path = f"snapshots/{snapshot_filename}"  # Path for serving via FastAPI
        full_snapshot_path = f"media/snapshots/{snapshot_filename}"  # Full path for file creation
        
        # Create the snapshot (mock image)
        self._create_mock_snapshot(full_snapshot_path, bbox_x, bbox_y, bbox_width, bbox_height, scenario.value)
        
        # Create alert
        alert = VideoAlert(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            confidence=confidence,
            source_pipeline="video_surveillance",
            status=AlertStatus.ACTIVE,
            event_type=scenario.value,
            bounding_box=[bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height],
            track_id=track_id,
            snapshot_path=snapshot_path,
            video_timestamp=datetime.now().isoformat(),
            metadata={
                'session_id': session_id,
                'location': location,
                'description': description,
                'mock_alert': True,
                'scenario_type': scenario.value,
                'detection_zone': f"Zone_{random.randint(1, 5)}",
                'camera_id': f"CAM_{random.randint(1, 10):02d}",
                'severity': self._get_severity_for_scenario(scenario)
            }
        )
        
        return alert
    
    def _get_severity_for_scenario(self, scenario: AlertScenario) -> str:
        """
        Get severity level for alert scenario
        
        Args:
            scenario: Alert scenario type
            
        Returns:
            Severity level string
        """
        severity_map = {
            AlertScenario.ZONE_VIOLATION: "high",
            AlertScenario.ABANDONED_OBJECT: "high",
            AlertScenario.LOITERING: "medium",
            AlertScenario.SUSPICIOUS_BEHAVIOR: "medium",
            AlertScenario.CROWD_DETECTION: "low"
        }
        
        return severity_map.get(scenario, "medium")
    
    async def _generate_alerts_loop(self, session_id: str):
        """
        Main loop for generating alerts at specified intervals
        
        Args:
            session_id: Analysis session ID
        """
        generator_data = self.active_generators[session_id]
        interval = generator_data['interval']
        
        try:
            logger.info(f"Starting alert generation loop for session {session_id}")
            
            while not generator_data['stop_event'].is_set():
                try:
                    # Wait for interval or stop signal
                    await asyncio.wait_for(
                        generator_data['stop_event'].wait(),
                        timeout=interval
                    )
                    # If we get here, stop was signaled
                    break
                    
                except asyncio.TimeoutError:
                    # Timeout reached, generate alert
                    try:
                        alert = self.generate_realistic_alert(session_id)
                        generator_data['alerts_generated'] += 1
                        
                        # Send alert through callback
                        if self.alert_callback:
                            logger.info(f"Sending alert {alert.id} through callback")
                            await self.alert_callback(alert)
                            logger.info(f"Alert {alert.id} sent successfully")
                        else:
                            logger.warning("No alert callback set!")
                        
                        logger.info(f"Generated mock alert for session {session_id}: {alert.event_type} (confidence: {alert.confidence})")
                        
                    except Exception as e:
                        logger.error(f"Error generating alert for session {session_id}: {str(e)}")
                        continue
            
            logger.info(f"Alert generation loop stopped for session {session_id}")
            
        except asyncio.CancelledError:
            logger.info(f"Alert generation cancelled for session {session_id}")
        except Exception as e:
            logger.error(f"Error in alert generation loop for session {session_id}: {str(e)}")
    
    def get_generator_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of mock alert generator for a session
        
        Args:
            session_id: Analysis session ID
            
        Returns:
            Generator status dict or None if not found
        """
        if session_id not in self.active_generators:
            return None
        
        generator_data = self.active_generators[session_id]
        
        return {
            'session_id': session_id,
            'active': not generator_data['stop_event'].is_set(),
            'interval': generator_data['interval'],
            'alerts_generated': generator_data['alerts_generated'],
            'started_at': generator_data['started_at'].isoformat(),
            'uptime_seconds': (datetime.now() - generator_data['started_at']).total_seconds()
        }
    
    def list_active_generators(self) -> List[Dict[str, Any]]:
        """
        Get list of all active alert generators
        
        Returns:
            List of generator status dicts
        """
        generators = []
        for session_id in self.active_generators:
            status = self.get_generator_status(session_id)
            if status:
                generators.append(status)
        return generators
    
    async def cleanup_generator(self, session_id: str):
        """
        Clean up alert generator resources
        
        Args:
            session_id: Analysis session ID
        """
        try:
            await self.stop_mock_alerts(session_id)
            logger.info(f"Cleaned up mock alert generator for session {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up generator for session {session_id}: {str(e)}")


    def _create_mock_snapshot(self, snapshot_path: str, bbox_x: int, bbox_y: int, bbox_width: int, bbox_height: int, event_type: str):
        """
        Create a mock snapshot image with bounding box
        
        Args:
            snapshot_path: Path where to save the snapshot
            bbox_x, bbox_y, bbox_width, bbox_height: Bounding box coordinates
            event_type: Type of event for the snapshot
        """
        try:
            import cv2
            import numpy as np
            import os
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            
            # Create a mock image (640x480 with some pattern)
            height, width = 480, 640
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add some background pattern
            image[:] = (50, 50, 50)  # Dark gray background
            
            # Add some random noise to make it look more realistic
            noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
            image = cv2.add(image, noise)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for normal detection
            if event_type == 'zone_violation':
                color = (0, 0, 255)  # Red for violations
            elif event_type == 'abandoned_object':
                color = (0, 255, 255)  # Yellow for abandoned objects
            
            # Ensure bounding box is within image bounds
            bbox_x = max(0, min(bbox_x, width - bbox_width))
            bbox_y = max(0, min(bbox_y, height - bbox_height))
            bbox_width = min(bbox_width, width - bbox_x)
            bbox_height = min(bbox_height, height - bbox_y)
            
            # Draw rectangle
            cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), color, 2)
            
            # Add label
            label = event_type.replace('_', ' ').title()
            cv2.putText(image, label, (bbox_x, bbox_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Save image
            cv2.imwrite(snapshot_path, image)
            logger.info(f"Created mock snapshot: {snapshot_path}")
            
        except Exception as e:
            logger.warning(f"Could not create mock snapshot {snapshot_path}: {str(e)}")
            # Create a placeholder file
            try:
                os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
                with open(snapshot_path, 'w') as f:
                    f.write("Mock snapshot placeholder")
            except:
                pass


# Global generator instance
mock_alert_generator = MockAlertGenerator()