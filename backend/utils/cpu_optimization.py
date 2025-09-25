#!/usr/bin/env python3
"""
CPU Optimization Utilities
Optimizations for i5 6th Gen CPU performance.
"""

import os
import psutil
import threading
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CPUOptimizer:
    """CPU optimization utilities for i5 6th Gen processors."""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        self.logical_cpu_count = psutil.cpu_count(logical=True)  # Logical cores
        self.optimized = False
    
    def optimize_for_i5_6th_gen(self) -> Dict[str, Any]:
        """Apply optimizations specific to i5 6th Gen (4 cores, 4 threads)."""
        optimizations = {}
        
        # Set OpenMP thread limits
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        os.environ['NUMEXPR_NUM_THREADS'] = '4'
        optimizations['thread_limits'] = 4
        
        # Set CPU affinity for better performance
        try:
            import psutil
            current_process = psutil.Process()
            # Use all available cores
            current_process.cpu_affinity(list(range(self.logical_cpu_count)))
            optimizations['cpu_affinity'] = list(range(self.logical_cpu_count))
        except Exception as e:
            logger.warning(f"Could not set CPU affinity: {e}")
        
        # Memory optimization
        os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'
        optimizations['memory_optimization'] = True
        
        self.optimized = True
        logger.info("Applied i5 6th Gen CPU optimizations")
        return optimizations
    
    def get_optimal_batch_size(self, base_batch_size: int = 8) -> int:
        """Calculate optimal batch size based on available memory."""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < 4:
            return max(1, base_batch_size // 4)
        elif available_gb < 8:
            return max(2, base_batch_size // 2)
        else:
            return base_batch_size
    
    def get_optimal_worker_count(self) -> int:
        """Get optimal number of worker processes."""
        return min(4, self.cpu_count)  # Max 4 for i5 6th gen
    
    def monitor_performance(self) -> Dict[str, float]:
        """Monitor current CPU and memory usage."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }

# Global optimizer instance
cpu_optimizer = CPUOptimizer()