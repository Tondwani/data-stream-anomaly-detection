from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Tuple
import time
import numpy as np

# Performance metrics tracking
@dataclass
class PerformanceMetrics:
    """Stores and calculates performance metrics."""
    processing_times: List[float] = None
    detection_times: List[float] = None
    memory_usage: List[float] = None
    
    def __post_init__(self):
        self.processing_times = []
        self.detection_times = []
        self.memory_usage = []
    
    def add_metric(self, processing_time: float, detection_time: float, memory: float):
        """Add new metrics."""
        self.processing_times.append(processing_time)
        self.detection_times.append(detection_time)
        self.memory_usage.append(memory)
    
    def get_summary(self) -> Dict[str, float]:
        """Calculate summary statistics."""
        return {
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
            'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0,
            'max_processing_time': max(self.processing_times) if self.processing_times else 0,
            'max_detection_time': max(self.detection_times) if self.detection_times else 0,
            'max_memory_usage': max(self.memory_usage) if self.memory_usage else 0
        }

# Base class for anomaly detection algorithms
class AnomalyDetectorBase(ABC):
    """Abstract base class for anomaly detection algorithms."""
    
    def __init__(self, config: dict):
        self.config = config
        self.metrics = PerformanceMetrics()
    
    @abstractmethod
    def is_anomaly(self, value: float) -> Tuple[bool, float]:
        """Detect if a value is anomalous."""
        pass

class ZScoreDetector(AnomalyDetectorBase):
    """Z-score based anomaly detection."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.window_size = config['detection']['window_size']
        self.z_threshold = config['detection']['z_score_threshold']
        self.values = deque(maxlen=self.window_size)
        self.lock = Lock()
    
    def is_anomaly(self, value: float) -> Tuple[bool, float]:
        start_time = time.time()
        with self.lock:
            self.values.append(value)
            if len(self.values) >= 2:
                mean = np.mean(self.values)
                std = np.std(self.values) + 1e-8
                z_score = abs(value - mean) / std
                is_anomalous = z_score > self.z_threshold
            else:
                z_score = 0
                is_anomalous = False
        
        detection_time = time.time() - start_time
        self.metrics.add_metric(detection_time, detection_time, len(self.values))
        return is_anomalous, z_score

class EWMADetector(AnomalyDetectorBase):
    """Exponentially Weighted Moving Average detector."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.alpha = config['detection']['ewma_alpha']
        self.ewma = None
        self.ewmvar = None
        self.lock = Lock()
    
    def is_anomaly(self, value: float) -> Tuple[bool, float]:
        start_time = time.time()
        with self.lock:
            if self.ewma is None:
                self.ewma = value
                self.ewmvar = 0
                score = 0
                is_anomalous = False
            else:
                # Update EWMA statistics
                diff = value - self.ewma
                incr = self.alpha * diff
                self.ewma += incr
                self.ewmvar = (1 - self.alpha) * (self.ewmvar + self.alpha * diff * diff)
                
                # Calculate score
                score = abs(diff) / (np.sqrt(self.ewmvar) + 1e-8)
                is_anomalous = score > self.config['detection']['z_score_threshold']
        
        detection_time = time.time() - start_time
        self.metrics.add_metric(detection_time, detection_time, 0)
        return is_anomalous, score

class RobustZScoreDetector(AnomalyDetectorBase):
    """Robust Z-score detector using median and MAD."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.window_size = config['detection']['window_size']
        self.threshold = config['detection']['robust_zscore_threshold']
        self.values = deque(maxlen=self.window_size)
        self.lock = Lock()
    
    def is_anomaly(self, value: float) -> Tuple[bool, float]:
        start_time = time.time()
        with self.lock:
            self.values.append(value)
            if len(self.values) >= 2:
                median = np.median(self.values)
                mad = np.median(np.abs(np.array(self.values) - median))
                score = abs(value - median) / (mad + 1e-8)
                is_anomalous = score > self.threshold
            else:
                score = 0
                is_anomalous = False
        
        detection_time = time.time() - start_time
        self.metrics.add_metric(detection_time, detection_time, len(self.values))
        return is_anomalous, score