#!/usr/bin/env python3
"""
HifazatAI Demo Pipeline
Creates demonstration data and workflows for the complete system.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoDataGenerator:
    """Generates realistic demo data for all pipelines."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.demo_running = False
    
    def generate_threat_alert(self) -> Dict[str, Any]:
        """Generate a realistic threat intelligence alert."""
        threat_types = [
            {
                "ioc_type": "ip",
                "ioc_value": f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
                "risk_level": "High",
                "evidence_text": "Detected communication with known botnet C&C server",
                "source_feed": "ThreatIntel_Feed_1"
            },
            {
                "ioc_type": "domain",
                "ioc_value": f"malicious-{random.randint(1000, 9999)}.com",
                "risk_level": "High",
                "evidence_text": "Domain associated with phishing campaign",
                "source_feed": "ThreatIntel_Feed_2"
            },
            {
                "ioc_type": "hash",
                "ioc_value": f"{''.join(random.choices('0123456789abcdef', k=32))}",
                "risk_level": "Medium",
                "evidence_text": "Suspicious file hash detected in network traffic",
                "source_feed": "ThreatIntel_Feed_3"
            }
        ]
        
        threat = random.choice(threat_types)
        
        return {
            "id": f"threat_{int(time.time())}_{random.randint(1000, 9999)}",
            "timestamp": datetime.now().isoformat(),
            "confidence": random.uniform(0.7, 0.95),
            "source_pipeline": "threat_intelligence",
            "status": "active",
            **threat
        }
    
    def generate_video_alert(self) -> Dict[str, Any]:
        """Generate a realistic video surveillance alert."""
        event_types = [
            {
                "event_type": "loitering",
                "description": "Person detected loitering in restricted area"
            },
            {
                "event_type": "zone_violation", 
                "description": "Unauthorized entry into secure zone"
            },
            {
                "event_type": "abandoned_object",
                "description": "Unattended object detected"
            }
        ]
        
        event = random.choice(event_types)
        camera_id = random.randint(1, 5)
        track_id = random.randint(100, 999)
        
        # Generate realistic bounding box
        x = random.randint(50, 400)
        y = random.randint(50, 300)
        w = random.randint(80, 150)
        h = random.randint(120, 200)
        
        return {
            "id": f"video_{int(time.time())}_{random.randint(1000, 9999)}",
            "timestamp": datetime.now().isoformat(),
            "confidence": random.uniform(0.8, 0.95),
            "source_pipeline": "video_surveillance",
            "status": "active",
            "event_type": event["event_type"],
            "bounding_box": [x, y, x + w, y + h],
            "track_id": track_id,
            "snapshot_path": f"/snapshots/camera_{camera_id}_track_{track_id}.jpg",
            "video_timestamp": random.uniform(0, 300)
        }
    
    def generate_anomaly_alert(self) -> Dict[str, Any]:
        """Generate a realistic border anomaly alert."""
        anomaly_types = [
            {
                "anomaly_type": "erratic_movement",
                "description": "Highly irregular movement pattern detected"
            },
            {
                "anomaly_type": "unusual_speed",
                "description": "Movement speed significantly above normal"
            },
            {
                "anomaly_type": "suspicious_loitering",
                "description": "Extended presence in sensitive border area"
            },
            {
                "anomaly_type": "direction_anomaly",
                "description": "Movement against typical traffic flow"
            }
        ]
        
        anomaly = random.choice(anomaly_types)
        
        # Generate trajectory points
        trajectory_points = []
        start_x, start_y = random.randint(100, 500), random.randint(100, 400)
        
        for i in range(random.randint(10, 25)):
            # Add some randomness to create realistic trajectory
            start_x += random.randint(-20, 20)
            start_y += random.randint(-20, 20)
            trajectory_points.append([start_x, start_y])
        
        return {
            "id": f"anomaly_{int(time.time())}_{random.randint(1000, 9999)}",
            "timestamp": datetime.now().isoformat(),
            "confidence": random.uniform(0.6, 0.9),
            "source_pipeline": "border_anomaly",
            "status": "active",
            "anomaly_type": anomaly["anomaly_type"],
            "severity_score": random.uniform(0.6, 0.95),
            "trajectory_points": trajectory_points,
            "feature_vector": [random.uniform(0, 1) for _ in range(6)],
            "supporting_frames": [
                f"/frames/anomaly_{i}.jpg" for i in range(random.randint(2, 5))
            ]
        }
    
    async def send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert to the API."""
        try:
            pipeline = alert_data["source_pipeline"]
            endpoint_map = {
                "threat_intelligence": "/alerts/threat",
                "video_surveillance": "/alerts/video", 
                "border_anomaly": "/alerts/anomaly"
            }
            
            endpoint = endpoint_map.get(pipeline)
            if not endpoint:
                logger.error(f"Unknown pipeline: {pipeline}")
                return False
            
            url = f"{self.api_base_url}{endpoint}"
            
            # Use requests for synchronous HTTP call
            response = requests.post(url, json=alert_data, timeout=10)
            
            if response.status_code in [200, 201]:
                logger.info(f"✅ Sent {pipeline} alert: {alert_data['id']}")
                return True
            else:
                logger.error(f"❌ Failed to send alert: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending alert: {e}")
            return False
    
    async def run_demo_scenario(self, scenario_name: str = "mixed_threats"):
        """Run a specific demo scenario."""
        logger.info(f"🎬 Starting demo scenario: {scenario_name}")
        
        scenarios = {
            "mixed_threats": self._mixed_threats_scenario,
            "security_breach": self._security_breach_scenario,
            "border_incident": self._border_incident_scenario,
            "continuous_monitoring": self._continuous_monitoring_scenario
        }
        
        scenario_func = scenarios.get(scenario_name)
        if not scenario_func:
            logger.error(f"Unknown scenario: {scenario_name}")
            return
        
        self.demo_running = True
        try:
            await scenario_func()
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        finally:
            self.demo_running = False
            logger.info("🎬 Demo scenario completed")
    
    async def _mixed_threats_scenario(self):
        """Demo scenario with mixed threat types."""
        logger.info("Scenario: Mixed security threats over 10 minutes")
        
        # Generate alerts over 10 minutes
        for minute in range(10):
            logger.info(f"📅 Demo minute {minute + 1}/10")
            
            # Generate 2-5 alerts per minute
            num_alerts = random.randint(2, 5)
            
            for _ in range(num_alerts):
                # Random alert type
                alert_type = random.choice(["threat", "video", "anomaly"])
                
                if alert_type == "threat":
                    alert = self.generate_threat_alert()
                elif alert_type == "video":
                    alert = self.generate_video_alert()
                else:
                    alert = self.generate_anomaly_alert()
                
                await self.send_alert(alert)
                
                # Wait between alerts
                await asyncio.sleep(random.uniform(5, 15))
            
            if not self.demo_running:
                break
    
    async def _security_breach_scenario(self):
        """Demo scenario simulating a security breach."""
        logger.info("Scenario: Coordinated security breach simulation")
        
        # Phase 1: Initial threat intelligence
        logger.info("🚨 Phase 1: Threat intelligence alerts")
        for _ in range(3):
            alert = self.generate_threat_alert()
            alert["risk_level"] = "High"
            await self.send_alert(alert)
            await asyncio.sleep(10)
        
        # Phase 2: Video surveillance detections
        logger.info("🚨 Phase 2: Video surveillance alerts")
        for _ in range(4):
            alert = self.generate_video_alert()
            alert["event_type"] = "zone_violation"
            await self.send_alert(alert)
            await asyncio.sleep(8)
        
        # Phase 3: Border anomalies
        logger.info("🚨 Phase 3: Border anomaly alerts")
        for _ in range(2):
            alert = self.generate_anomaly_alert()
            alert["severity_score"] = random.uniform(0.8, 0.95)
            await self.send_alert(alert)
            await asyncio.sleep(12)
    
    async def _border_incident_scenario(self):
        """Demo scenario focused on border security."""
        logger.info("Scenario: Border security incident")
        
        # Generate coordinated border anomalies
        for i in range(8):
            alert = self.generate_anomaly_alert()
            
            # Make alerts more severe over time
            alert["severity_score"] = min(0.95, 0.6 + (i * 0.05))
            
            if i > 3:
                alert["anomaly_type"] = "erratic_movement"
            
            await self.send_alert(alert)
            await asyncio.sleep(random.uniform(15, 30))
    
    async def _continuous_monitoring_scenario(self):
        """Demo scenario for continuous monitoring."""
        logger.info("Scenario: Continuous monitoring (runs until stopped)")
        
        while self.demo_running:
            # Generate random alert
            alert_type = random.choices(
                ["threat", "video", "anomaly"],
                weights=[0.3, 0.5, 0.2]  # More video alerts
            )[0]
            
            if alert_type == "threat":
                alert = self.generate_threat_alert()
            elif alert_type == "video":
                alert = self.generate_video_alert()
            else:
                alert = self.generate_anomaly_alert()
            
            await self.send_alert(alert)
            
            # Wait 30-120 seconds between alerts
            await asyncio.sleep(random.uniform(30, 120))
    
    def stop_demo(self):
        """Stop the running demo."""
        self.demo_running = False

class DemoController:
    """Controls demo workflows and provides user interface."""
    
    def __init__(self):
        self.generator = DemoDataGenerator()
        self.demo_stats = {
            "alerts_sent": 0,
            "start_time": None,
            "scenarios_run": []
        }
    
    async def run_interactive_demo(self):
        """Run interactive demo with user choices."""
        print("🎬 HifazatAI Demo Controller")
        print("=" * 40)
        
        while True:
            print("\nAvailable Demo Scenarios:")
            print("1. Mixed Threats (10 minutes)")
            print("2. Security Breach Simulation")
            print("3. Border Incident")
            print("4. Continuous Monitoring")
            print("5. Single Alert Test")
            print("6. View Demo Statistics")
            print("0. Exit")
            
            try:
                choice = input("\nSelect scenario (0-6): ").strip()
                
                if choice == "0":
                    break
                elif choice == "1":
                    await self.generator.run_demo_scenario("mixed_threats")
                elif choice == "2":
                    await self.generator.run_demo_scenario("security_breach")
                elif choice == "3":
                    await self.generator.run_demo_scenario("border_incident")
                elif choice == "4":
                    await self.generator.run_demo_scenario("continuous_monitoring")
                elif choice == "5":
                    await self.send_single_alert()
                elif choice == "6":
                    self.show_demo_stats()
                else:
                    print("Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\nDemo interrupted. Stopping...")
                self.generator.stop_demo()
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def send_single_alert(self):
        """Send a single test alert."""
        print("\nSelect alert type:")
        print("1. Threat Intelligence")
        print("2. Video Surveillance") 
        print("3. Border Anomaly")
        
        choice = input("Select type (1-3): ").strip()
        
        if choice == "1":
            alert = self.generator.generate_threat_alert()
        elif choice == "2":
            alert = self.generator.generate_video_alert()
        elif choice == "3":
            alert = self.generator.generate_anomaly_alert()
        else:
            print("Invalid choice.")
            return
        
        success = await self.generator.send_alert(alert)
        if success:
            self.demo_stats["alerts_sent"] += 1
            print(f"✅ Alert sent successfully: {alert['id']}")
        else:
            print("❌ Failed to send alert")
    
    def show_demo_stats(self):
        """Show demo statistics."""
        print("\n📊 Demo Statistics")
        print("-" * 20)
        print(f"Alerts sent: {self.demo_stats['alerts_sent']}")
        print(f"Scenarios run: {len(self.demo_stats['scenarios_run'])}")
        if self.demo_stats['start_time']:
            runtime = datetime.now() - self.demo_stats['start_time']
            print(f"Runtime: {runtime}")

async def main():
    """Main demo entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='HifazatAI Demo Pipeline')
    parser.add_argument('--scenario', choices=[
        'mixed_threats', 'security_breach', 'border_incident', 'continuous_monitoring'
    ], help='Run specific scenario')
    parser.add_argument('--api-url', default='http://localhost:8000',
                       help='API base URL')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive demo')
    
    args = parser.parse_args()
    
    controller = DemoController()
    controller.generator.api_base_url = args.api_url
    
    if args.interactive:
        await controller.run_interactive_demo()
    elif args.scenario:
        await controller.generator.run_demo_scenario(args.scenario)
    else:
        # Default: run interactive demo
        await controller.run_interactive_demo()

if __name__ == "__main__":
    asyncio.run(main())