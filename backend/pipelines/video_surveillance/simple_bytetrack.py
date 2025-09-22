"""
Simplified ByteTrack implementation for CPU-optimized tracking.
Based on ByteTrack algorithm but without heavy dependencies.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
import logging
from datetime import datetime

from utils.logging import get_pipeline_logger

logger = get_pipeline_logger("simple_bytetrack")


class TrackState:
    """Track state enumeration"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Base track class"""
    
    _count = 0
    
    def __init__(self):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)
    
    @property
    def end_frame(self):
        return self.frame_id
    
    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count
    
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.generate_results(frame_id)
        
    def predict(self):
        """Predict next state"""
        if self.state != TrackState.Tracked:
            self.mean[7] = 0
        self.kalman_filter.predict(self.mean, self.covariance)
    
    def generate_results(self, frame_id):
        """Generate tracking results"""
        self.frame_id = frame_id
        self.state = TrackState.Tracked
        
    def update(self, new_track, frame_id):
        """Update track with new detection"""
        self.frame_id = frame_id
        self.state = TrackState.Tracked
        
        self.is_activated = True
        self.score = new_track.score
        
    def mark_lost(self):
        """Mark track as lost"""
        self.state = TrackState.Lost
        
    def mark_removed(self):
        """Mark track as removed"""
        self.state = TrackState.Removed


class STrack(BaseTrack):
    """Simple track implementation"""
    
    def __init__(self, tlwh, score, class_id=0):
        """
        Initialize track
        
        Args:
            tlwh: Bounding box in [top, left, width, height] format
            score: Detection confidence score
            class_id: Object class ID
        """
        super().__init__()
        
        # Convert tlwh to tlbr format
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        
        self.score = score
        self.class_id = class_id
        self.tracklet_len = 0
        
    @property
    def tlwh(self):
        """Get current position in tlwh format"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    @property
    def tlbr(self):
        """Get current position in tlbr format"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert tlwh to xyah format"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    def to_xyah(self):
        """Convert to xyah format"""
        return self.tlwh_to_xyah(self.tlwh)
    
    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Convert tlbr to tlwh format"""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """Convert tlwh to tlbr format"""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret
    
    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class SimpleKalmanFilter:
    """Simplified Kalman filter for object tracking"""
    
    def __init__(self):
        ndim, dt = 4, 1.0
        
        # Create Kalman filter matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    
    def initiate(self, measurement):
        """Create track from unassociated measurement"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step"""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def project(self, mean, covariance):
        """Project state distribution to measurement space"""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov
    
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step"""
        projected_mean, projected_cov = self.project(mean, covariance)
        
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class SimpleBYTETracker:
    """Simplified ByteTrack implementation"""
    
    def __init__(self, frame_rate=30, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        """
        Initialize ByteTracker
        
        Args:
            frame_rate: Video frame rate
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to buffer tracks
            match_thresh: Matching threshold for track association
        """
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.kalman_filter = SimpleKalmanFilter()
        
        logger.info("SimpleBYTETracker initialized")
    
    def update(self, output_results, img_info=None, img_size=None):
        """
        Update tracker with new detections
        
        Args:
            output_results: Detection results in format [[x1, y1, x2, y2, conf, class_id], ...]
            img_info: Image info (unused in simple implementation)
            img_size: Image size (unused in simple implementation)
            
        Returns:
            List of active tracks
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = np.zeros(len(bboxes))  # Default class 0
        else:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = output_results[:, 5] if output_results.shape[1] > 5 else np.zeros(len(bboxes))
        
        # Filter by confidence threshold
        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh
        
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        classes_keep = classes[remain_inds]
        classes_second = classes[inds_second]
        
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                         (tlbr, s, c) in zip(dets, scores_keep, classes_keep)]
        else:
            detections = []
        
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # Predict the current location with KF
        for strack in strack_pool:
            strack.predict()
        
        dists = matching.iou_distance(strack_pool, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                               (tlbr, s, c) in zip(dets_second, scores_second, classes_second)]
        else:
            detections_second = []
        
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)
        
        # print('===========Frame {}=========='.format(self.frame_id))
        # print('Activated: {}'.format([track.track_id for track in activated_starcks]))
        # print('Refind: {}'.format([track.track_id for track in refind_stracks]))
        # print('Lost: {}'.format([track.track_id for track in lost_stracks]))
        # print('Removed: {}'.format([track.track_id for track in removed_stracks]))
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        return output_stracks


def joint_stracks(tlista, tlistb):
    """Join two track lists"""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """Subtract track list b from track list a"""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """Remove duplicate tracks"""
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


# Simple matching functions
class matching:
    """Simple matching utilities"""
    
    @staticmethod
    def iou_distance(atracks, btracks):
        """Compute IoU distance matrix"""
        if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.tlbr for track in atracks]
            btlbrs = [track.tlbr for track in btracks]
        
        _ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
        if _ious.size == 0:
            return _ious
        
        for i, atlbr in enumerate(atlbrs):
            for j, btlbr in enumerate(btlbrs):
                _ious[i, j] = bbox_iou(atlbr, btlbr)
        
        cost_matrix = 1 - _ious
        return cost_matrix
    
    @staticmethod
    def fuse_score(cost_matrix, detections):
        """Fuse detection scores with cost matrix"""
        if cost_matrix.size == 0:
            return cost_matrix
        iou_sim = 1 - cost_matrix
        det_scores = np.array([det.score for det in detections])
        det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
        fuse_sim = iou_sim * det_scores
        fuse_cost = 1 - fuse_sim
        return fuse_cost
    
    @staticmethod
    def linear_assignment(cost_matrix, thresh):
        """Simple linear assignment using greedy matching"""
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        
        matches, unmatched_a, unmatched_b = [], [], []
        
        # Simple greedy matching
        cost_matrix_copy = cost_matrix.copy()
        
        while True:
            # Find minimum cost
            min_cost = np.min(cost_matrix_copy)
            if min_cost > thresh:
                break
            
            # Find indices of minimum cost
            min_indices = np.where(cost_matrix_copy == min_cost)
            if len(min_indices[0]) == 0:
                break
            
            i, j = min_indices[0][0], min_indices[1][0]
            matches.append([i, j])
            
            # Remove matched row and column
            cost_matrix_copy[i, :] = thresh + 1
            cost_matrix_copy[:, j] = thresh + 1
        
        # Find unmatched
        matched_a = [m[0] for m in matches]
        matched_b = [m[1] for m in matches]
        
        for i in range(cost_matrix.shape[0]):
            if i not in matched_a:
                unmatched_a.append(i)
        
        for j in range(cost_matrix.shape[1]):
            if j not in matched_b:
                unmatched_b.append(j)
        
        return np.array(matches), unmatched_a, unmatched_b


def bbox_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union