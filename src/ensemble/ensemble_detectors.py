from threading import Lock
from typing import Dict, Tuple

from ..anomaly_detection.detectors import (EWMADetector, RobustZScoreDetector,
                                           ZScoreDetector)


class DetectorEnsemble:
    """Ensemble of multiple anomaly detectors."""
    
    def __init__(self, config: dict):
        self.detectors = {
            'zscore': ZScoreDetector(config),
            'ewma': EWMADetector(config),
            'robust_zscore': RobustZScoreDetector(config)
        }
        self.lock = Lock()
    
    def check_anomaly(self, value: float) -> Dict[str, Tuple[bool, float]]:
        """Check for anomalies using all detectors."""
        with self.lock:
            results = {}
            for name, detector in self.detectors.items():
                is_anomaly, score = detector.is_anomaly(value)
                results[name] = (is_anomaly, score)
            return results