"""
Trajectory extraction and analysis for border anomaly detection.
Processes object tracking data to compute movement features and patterns.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from collections import defaultdict

from utils.logging import get_pipeline_logger

logger = get_pipeline_logger("border_anomaly")


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory"""
    x: float
    y: float
    timestamp: datetime
    frame_number: int
    confidence: float = 1.0


@dataclass
class Trajectory:
    """Complete trajectory of an object"""
    track_id: int
    points: List[TrajectoryPoint]
    start_time: datetime
    end_time: datetime
    object_class: str = "unknown"
    
    @property
    def duration(self) -> float:
        """Duration of trajectory in seconds"""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def length(self) -> int:
        """Number of points in trajectory"""
        return len(self.points)
    
    @property
    def is_valid(self) -> bool:
        """Check if trajectory has minimum required points"""
        return len(self.points) >= 3


@dataclass
class TrajectoryFeatures:
    """Computed features for a trajectory"""
    track_id: int
    
    # Basic metrics
    total_distance: float
    displacement: float
    duration: float
    average_speed: float
    max_speed: float
    
    # Path characteristics
    path_curvature: float
    direction_changes: int
    straightness_ratio: float
    
    # Spatial features
    entry_angle: float
    exit_angle: float
    bounding_box_area: float
    path_complexity: float
    
    # Temporal features
    acceleration_variance: float
    stop_duration: float
    movement_consistency: float


class TrajectoryExtractor:
    """Extracts trajectories from object tracking data"""
    
    def __init__(self, min_trajectory_length: int = 5, max_gap_frames: int = 10):
        """
        Initialize trajectory extractor
        
        Args:
            min_trajectory_length: Minimum number of points for valid trajectory
            max_gap_frames: Maximum frame gap to allow in trajectory
        """
        self.min_trajectory_length = min_trajectory_length
        self.max_gap_frames = max_gap_frames
        self.active_trajectories: Dict[int, List[TrajectoryPoint]] = defaultdict(list)
        self.completed_trajectories: List[Trajectory] = []
        
        logger.info(f"TrajectoryExtractor initialized (min_length={min_trajectory_length}, max_gap={max_gap_frames})")
    
    def update(self, tracked_objects: List[Dict[str, Any]], frame_number: int, timestamp: datetime) -> List[Trajectory]:
        """
        Update trajectories with new tracking data
        
        Args:
            tracked_objects: List of tracked objects from video pipeline
            frame_number: Current frame number
            timestamp: Current timestamp
            
        Returns:
            List of newly completed trajectories
        """
        current_track_ids = set()
        new_completed = []
        
        # Process current detections
        for obj in tracked_objects:
            track_id = obj.get('track_id')
            if track_id is None:
                continue
                
            current_track_ids.add(track_id)
            
            # Extract position from bounding box center
            bbox = obj.get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
            else:
                center_x, center_y = obj.get('center', [0, 0])
            
            # Create trajectory point
            point = TrajectoryPoint(
                x=center_x,
                y=center_y,
                timestamp=timestamp,
                frame_number=frame_number,
                confidence=obj.get('confidence', 1.0)
            )
            
            # Add to active trajectory
            self.active_trajectories[track_id].append(point)
        
        # Check for completed trajectories (tracks that disappeared)
        completed_track_ids = []
        for track_id in list(self.active_trajectories.keys()):
            if track_id not in current_track_ids:
                # Check if trajectory has been inactive for too long
                last_point = self.active_trajectories[track_id][-1]
                gap_frames = frame_number - last_point.frame_number
                
                if gap_frames > self.max_gap_frames:
                    # Complete this trajectory
                    trajectory = self._create_trajectory(track_id)
                    if trajectory and trajectory.is_valid:
                        new_completed.append(trajectory)
                        self.completed_trajectories.append(trajectory)
                    
                    completed_track_ids.append(track_id)
        
        # Remove completed trajectories
        for track_id in completed_track_ids:
            del self.active_trajectories[track_id]
        
        logger.debug(f"Updated trajectories: {len(current_track_ids)} active, {len(new_completed)} completed")
        return new_completed
    
    def _create_trajectory(self, track_id: int) -> Optional[Trajectory]:
        """Create trajectory object from points"""
        points = self.active_trajectories.get(track_id, [])
        
        if len(points) < self.min_trajectory_length:
            return None
        
        # Sort points by timestamp
        points.sort(key=lambda p: p.timestamp)
        
        trajectory = Trajectory(
            track_id=track_id,
            points=points,
            start_time=points[0].timestamp,
            end_time=points[-1].timestamp,
            object_class="unknown"  # Could be enhanced with class info
        )
        
        return trajectory
    
    def get_active_trajectories(self) -> Dict[int, List[TrajectoryPoint]]:
        """Get currently active trajectories"""
        return dict(self.active_trajectories)
    
    def get_completed_trajectories(self) -> List[Trajectory]:
        """Get all completed trajectories"""
        return self.completed_trajectories.copy()
    
    def force_complete_all(self) -> List[Trajectory]:
        """Force completion of all active trajectories"""
        completed = []
        for track_id in list(self.active_trajectories.keys()):
            trajectory = self._create_trajectory(track_id)
            if trajectory and trajectory.is_valid:
                completed.append(trajectory)
                self.completed_trajectories.append(trajectory)
        
        self.active_trajectories.clear()
        return completed


class TrajectoryAnalyzer:
    """Analyzes trajectories to compute movement features"""
    
    def __init__(self, frame_width: int = 1920, frame_height: int = 1080, fps: float = 30.0):
        """
        Initialize trajectory analyzer
        
        Args:
            frame_width: Video frame width for normalization
            frame_height: Video frame height for normalization  
            fps: Video frame rate for speed calculations
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        logger.info(f"TrajectoryAnalyzer initialized ({frame_width}x{frame_height} @ {fps}fps)")
    
    def analyze_trajectory(self, trajectory: Trajectory) -> TrajectoryFeatures:
        """
        Analyze a trajectory and compute features
        
        Args:
            trajectory: Trajectory to analyze
            
        Returns:
            Computed trajectory features
        """
        if not trajectory.is_valid:
            raise ValueError("Invalid trajectory - insufficient points")
        
        points = trajectory.points
        
        # Basic distance and displacement calculations
        total_distance = self._calculate_total_distance(points)
        displacement = self._calculate_displacement(points)
        duration = trajectory.duration
        
        # Speed calculations
        speeds = self._calculate_speeds(points)
        average_speed = np.mean(speeds) if speeds else 0.0
        max_speed = np.max(speeds) if speeds else 0.0
        
        # Path characteristics
        path_curvature = self._calculate_curvature(points)
        direction_changes = self._count_direction_changes(points)
        straightness_ratio = displacement / total_distance if total_distance > 0 else 0.0
        
        # Spatial features
        entry_angle = self._calculate_entry_angle(points)
        exit_angle = self._calculate_exit_angle(points)
        bounding_box_area = self._calculate_bounding_box_area(points)
        path_complexity = self._calculate_path_complexity(points)
        
        # Temporal features
        accelerations = self._calculate_accelerations(points)
        acceleration_variance = np.var(accelerations) if accelerations else 0.0
        stop_duration = self._calculate_stop_duration(points)
        movement_consistency = self._calculate_movement_consistency(points)
        
        return TrajectoryFeatures(
            track_id=trajectory.track_id,
            total_distance=total_distance,
            displacement=displacement,
            duration=duration,
            average_speed=average_speed,
            max_speed=max_speed,
            path_curvature=path_curvature,
            direction_changes=direction_changes,
            straightness_ratio=straightness_ratio,
            entry_angle=entry_angle,
            exit_angle=exit_angle,
            bounding_box_area=bounding_box_area,
            path_complexity=path_complexity,
            acceleration_variance=acceleration_variance,
            stop_duration=stop_duration,
            movement_consistency=movement_consistency
        )
    
    def _calculate_total_distance(self, points: List[TrajectoryPoint]) -> float:
        """Calculate total distance traveled along trajectory"""
        if len(points) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(points)):
            dx = points[i].x - points[i-1].x
            dy = points[i].y - points[i-1].y
            total_distance += math.sqrt(dx*dx + dy*dy)
        
        return total_distance
    
    def _calculate_displacement(self, points: List[TrajectoryPoint]) -> float:
        """Calculate straight-line displacement from start to end"""
        if len(points) < 2:
            return 0.0
        
        start = points[0]
        end = points[-1]
        dx = end.x - start.x
        dy = end.y - start.y
        
        return math.sqrt(dx*dx + dy*dy)
    
    def _calculate_speeds(self, points: List[TrajectoryPoint]) -> List[float]:
        """Calculate instantaneous speeds between consecutive points"""
        if len(points) < 2:
            return []
        
        speeds = []
        for i in range(1, len(points)):
            dx = points[i].x - points[i-1].x
            dy = points[i].y - points[i-1].y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate time difference in seconds
            time_diff = (points[i].timestamp - points[i-1].timestamp).total_seconds()
            if time_diff > 0:
                speed = distance / time_diff  # pixels per second
                speeds.append(speed)
        
        return speeds
    
    def _calculate_curvature(self, points: List[TrajectoryPoint]) -> float:
        """Calculate average path curvature"""
        if len(points) < 3:
            return 0.0
        
        curvatures = []
        for i in range(1, len(points) - 1):
            # Calculate vectors
            v1_x = points[i].x - points[i-1].x
            v1_y = points[i].y - points[i-1].y
            v2_x = points[i+1].x - points[i].x
            v2_y = points[i+1].y - points[i].y
            
            # Calculate angle between vectors
            dot_product = v1_x * v2_x + v1_y * v2_y
            mag1 = math.sqrt(v1_x*v1_x + v1_y*v1_y)
            mag2 = math.sqrt(v2_x*v2_x + v2_y*v2_y)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
                angle = math.acos(cos_angle)
                curvatures.append(angle)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def _count_direction_changes(self, points: List[TrajectoryPoint], threshold: float = 0.5) -> int:
        """Count significant direction changes in trajectory"""
        if len(points) < 3:
            return 0
        
        direction_changes = 0
        prev_direction = None
        
        for i in range(1, len(points)):
            dx = points[i].x - points[i-1].x
            dy = points[i].y - points[i-1].y
            
            if abs(dx) > threshold or abs(dy) > threshold:
                current_direction = math.atan2(dy, dx)
                
                if prev_direction is not None:
                    angle_diff = abs(current_direction - prev_direction)
                    # Normalize angle difference to [0, pi]
                    angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                    
                    if angle_diff > math.pi/4:  # 45 degrees threshold
                        direction_changes += 1
                
                prev_direction = current_direction
        
        return direction_changes
    
    def _calculate_entry_angle(self, points: List[TrajectoryPoint]) -> float:
        """Calculate angle of entry into frame"""
        if len(points) < 2:
            return 0.0
        
        # Use first few points to determine entry direction
        start_idx = 0
        end_idx = min(3, len(points) - 1)
        
        dx = points[end_idx].x - points[start_idx].x
        dy = points[end_idx].y - points[start_idx].y
        
        return math.atan2(dy, dx)
    
    def _calculate_exit_angle(self, points: List[TrajectoryPoint]) -> float:
        """Calculate angle of exit from frame"""
        if len(points) < 2:
            return 0.0
        
        # Use last few points to determine exit direction
        start_idx = max(0, len(points) - 4)
        end_idx = len(points) - 1
        
        dx = points[end_idx].x - points[start_idx].x
        dy = points[end_idx].y - points[start_idx].y
        
        return math.atan2(dy, dx)
    
    def _calculate_bounding_box_area(self, points: List[TrajectoryPoint]) -> float:
        """Calculate area of trajectory bounding box"""
        if not points:
            return 0.0
        
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        return width * height
    
    def _calculate_path_complexity(self, points: List[TrajectoryPoint]) -> float:
        """Calculate path complexity as ratio of actual path to bounding box diagonal"""
        if len(points) < 2:
            return 0.0
        
        total_distance = self._calculate_total_distance(points)
        
        # Calculate bounding box diagonal
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        diagonal = math.sqrt(width*width + height*height)
        
        if diagonal > 0:
            return total_distance / diagonal
        else:
            return 0.0
    
    def _calculate_accelerations(self, points: List[TrajectoryPoint]) -> List[float]:
        """Calculate accelerations between consecutive speed measurements"""
        speeds = self._calculate_speeds(points)
        if len(speeds) < 2:
            return []
        
        accelerations = []
        for i in range(1, len(speeds)):
            # Time difference between speed measurements
            time_diff = (points[i+1].timestamp - points[i].timestamp).total_seconds()
            if time_diff > 0:
                acceleration = (speeds[i] - speeds[i-1]) / time_diff
                accelerations.append(acceleration)
        
        return accelerations
    
    def _calculate_stop_duration(self, points: List[TrajectoryPoint], speed_threshold: float = 5.0) -> float:
        """Calculate total time spent stationary or moving very slowly"""
        if len(points) < 2:
            return 0.0
        
        speeds = self._calculate_speeds(points)
        stop_duration = 0.0
        
        for i, speed in enumerate(speeds):
            if speed < speed_threshold:
                # Add time interval for this slow/stopped segment
                time_diff = (points[i+1].timestamp - points[i].timestamp).total_seconds()
                stop_duration += time_diff
        
        return stop_duration
    
    def _calculate_movement_consistency(self, points: List[TrajectoryPoint]) -> float:
        """Calculate consistency of movement (lower variance = more consistent)"""
        speeds = self._calculate_speeds(points)
        if len(speeds) < 2:
            return 1.0
        
        speed_variance = np.var(speeds)
        mean_speed = np.mean(speeds)
        
        if mean_speed > 0:
            # Coefficient of variation (normalized consistency measure)
            cv = math.sqrt(speed_variance) / mean_speed
            # Convert to consistency score (higher = more consistent)
            consistency = 1.0 / (1.0 + cv)
        else:
            consistency = 1.0
        
        return consistency


class TrajectoryVisualizer:
    """Visualizes trajectories for debugging and analysis"""
    
    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        """
        Initialize trajectory visualizer
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        logger.info(f"TrajectoryVisualizer initialized ({frame_width}x{frame_height})")
    
    def plot_trajectory(self, trajectory: Trajectory, title: str = None, save_path: str = None) -> None:
        """
        Plot a single trajectory
        
        Args:
            trajectory: Trajectory to plot
            title: Optional plot title
            save_path: Optional path to save plot
        """
        if not trajectory.points:
            logger.warning("Cannot plot empty trajectory")
            return
        
        x_coords = [p.x for p in trajectory.points]
        y_coords = [p.y for p in trajectory.points]
        
        plt.figure(figsize=(12, 8))
        
        # Plot trajectory path
        plt.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='Path')
        
        # Mark start and end points
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
        plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
        
        # Add direction arrows
        self._add_direction_arrows(x_coords, y_coords)
        
        plt.xlim(0, self.frame_width)
        plt.ylim(0, self.frame_height)
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.title(title or f'Trajectory {trajectory.track_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Trajectory plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_multiple_trajectories(self, trajectories: List[Trajectory], title: str = None, save_path: str = None) -> None:
        """
        Plot multiple trajectories on the same figure
        
        Args:
            trajectories: List of trajectories to plot
            title: Optional plot title
            save_path: Optional path to save plot
        """
        if not trajectories:
            logger.warning("Cannot plot empty trajectory list")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Use different colors for each trajectory
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
        
        for i, trajectory in enumerate(trajectories):
            if not trajectory.points:
                continue
            
            x_coords = [p.x for p in trajectory.points]
            y_coords = [p.y for p in trajectory.points]
            color = colors[i]
            
            # Plot trajectory path
            plt.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7, 
                    label=f'Track {trajectory.track_id}')
            
            # Mark start point
            plt.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=8)
            
            # Mark end point
            plt.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=8)
        
        plt.xlim(0, self.frame_width)
        plt.ylim(0, self.frame_height)
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.title(title or f'Multiple Trajectories ({len(trajectories)} tracks)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Multi-trajectory plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _add_direction_arrows(self, x_coords: List[float], y_coords: List[float], num_arrows: int = 5) -> None:
        """Add direction arrows to trajectory plot"""
        if len(x_coords) < 2:
            return
        
        # Calculate arrow positions
        arrow_indices = np.linspace(1, len(x_coords) - 1, num_arrows, dtype=int)
        
        for i in arrow_indices:
            if i < len(x_coords) - 1:
                dx = x_coords[i+1] - x_coords[i-1]
                dy = y_coords[i+1] - y_coords[i-1]
                
                # Normalize arrow length
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx = dx / length * 20  # Arrow length in pixels
                    dy = dy / length * 20
                
                plt.arrow(x_coords[i], y_coords[i], dx, dy, 
                         head_width=10, head_length=15, fc='red', ec='red', alpha=0.7)


def create_sample_trajectory(track_id: int = 1, num_points: int = 20) -> Trajectory:
    """
    Create a sample trajectory for testing purposes
    
    Args:
        track_id: Track ID for the trajectory
        num_points: Number of points to generate
        
    Returns:
        Sample trajectory
    """
    import random
    
    # Generate sample trajectory points
    points = []
    start_time = datetime.now()
    
    # Start position
    x, y = 100, 100
    
    for i in range(num_points):
        # Add some random movement
        x += random.uniform(-20, 20)
        y += random.uniform(-20, 20)
        
        # Keep within reasonable bounds
        x = max(0, min(1920, x))
        y = max(0, min(1080, y))
        
        timestamp = start_time + timedelta(seconds=i * 0.5)
        
        point = TrajectoryPoint(
            x=x,
            y=y,
            timestamp=timestamp,
            frame_number=i,
            confidence=random.uniform(0.8, 1.0)
        )
        points.append(point)
    
    trajectory = Trajectory(
        track_id=track_id,
        points=points,
        start_time=points[0].timestamp,
        end_time=points[-1].timestamp,
        object_class="person"
    )
    
    return trajectory