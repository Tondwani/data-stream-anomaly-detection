import json
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from threading import Lock, Thread
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation


# Configuration management
class ConfigManager:
    """Manages configuration loading and validation."""
    
    DEFAULT_CONFIG = {
        'data_generation': {
            'base_seasonal_period': 24,
            'secondary_seasonal_period': 168,  # Weekly pattern
            'noise_level': 0.1,
            'anomaly_rate': 0.02,
            'trend_coefficient': 0.001
        },
        'detection': {
            'window_size': 100,
            'z_score_threshold': 3.0,
            'ewma_alpha': 0.1,
            'isolation_forest_samples': 256,
            'robust_zscore_threshold': 3.0
        },
        'visualization': {
            'max_points': 1000,
            'update_interval': 50
        }
    }
    
    @classmethod
    def load_config(cls, config_path: str = None) -> dict:
        """Load configuration from file or use defaults."""
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return cls._merge_with_defaults(config)
            except Exception as e:
                logging.warning(f"Failed to load config file: {e}. Using defaults.")
        return cls.DEFAULT_CONFIG
    
    @classmethod
    def _merge_with_defaults(cls, config: dict) -> dict:
        """Merge loaded config with defaults to ensure all required fields exist."""
        merged = cls.DEFAULT_CONFIG.copy()
        for key, value in config.items():
            if isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
        return merged

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
            'max_detection_time': max(self.detection_times) if self.detection_usage else 0,
            'max_memory_usage': max(self.memory_usage) if self.memory_usage else 0
        }

class DataStreamGenerator:
    """Enhanced data stream generator with multiple seasonal patterns."""
    
    def __init__(self, config: dict):
        self.config = config['data_generation']
        self.time = 0
        self.trends = []
        self._initialize_trends()
    
    def _initialize_trends(self):
        """Initialize multiple seasonal trends."""
        # Primary daily seasonality
        self.trends.append({
            'period': self.config['base_seasonal_period'],
            'amplitude': 1.0,
            'phase': 0
        })
        # Weekly seasonality
        self.trends.append({
            'period': self.config['secondary_seasonal_period'],
            'amplitude': 0.5,
            'phase': np.pi / 4
        })
        # Monthly trend (approximate)
        self.trends.append({
            'period': 720,
            'amplitude': 0.3,
            'phase': np.pi / 6
        })
    
    def next_value(self) -> float:
        """Generate next value with multiple seasonal patterns."""
        # Combine multiple seasonal patterns
        seasonal = sum(
            trend['amplitude'] * np.sin(2 * np.pi * self.time / trend['period'] + trend['phase'])
            for trend in self.trends
        )
        
        # Add trend and noise
        trend = self.config['trend_coefficient'] * self.time
        noise = np.random.normal(0, self.config['noise_level'])
        
        value = seasonal + trend + noise
        
        # Inject anomalies
        if np.random.random() < self.config['anomaly_rate']:
            anomaly_type = np.random.choice(['spike', 'level_shift', 'trend_change'])
            if anomaly_type == 'spike':
                value += np.random.choice([-1, 1]) * np.random.uniform(3, 5)
            elif anomaly_type == 'level_shift':
                value += np.random.choice([-1, 1]) * 2
            else:  # trend_change
                value += 0.1 * self.time
        
        self.time += 1
        return value

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
                is_anomaly, score, detection_time = detector.is_anomaly(value)
                results[name] = (is_anomaly, score, detection_time)
            return results

class RealTimeVisualizer: 
    """Enhanced real-time visualization with multiple detector results."""
    
    def __init__(self, config: dict):
        self.max_points = config['visualization']['max_points']
        self.times = deque(maxlen=self.max_points)
        self.values = deque(maxlen=self.max_points)
        self.anomalies = {
            'zscore': {'x': deque(maxlen=self.max_points), 'y': deque(maxlen=self.max_points)},
            'ewma': {'x': deque(maxlen=self.max_points), 'y': deque(maxlen=self.max_points)},
            'robust_zscore': {'x': deque(maxlen=self.max_points), 'y': deque(maxlen=self.max_points)}
        }
        
        # detection time tracking
        self.detection_times = {
            'zscore': deque(maxlen=self.max_points),
            'ewma': deque(maxlen=self.max_points),
            'robust_zscore': deque(maxlen=self.max_points)
        }

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Add this one line to fix the spacing
        plt.subplots_adjust(hspace=0.5)
        self.setup_plots()
    
    def setup_plots(self):
        """Initialize plot settings."""
        # Data stream plot
        self.line, = self.ax1.plot([], [], 'b-', label='Data Stream')
        colors = {'zscore': 'red', 'ewma': 'green', 'robust_zscore': 'purple'}
        self.scatter_plots = {}
        for name, color in colors.items():
            self.scatter_plots[name] = self.ax1.scatter([], [], 
                                                        color=color, 
                                                        label=f'{name} anomalies')
        
        self.ax1.legend()
        self.ax1.set_title('Real-time Data Stream with Anomalies')
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Value')
        
        # Performance metrics plot
        self.perf_lines = {}
        self.ax2.set_title('Detection Times')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Detection Time (s)')
    
    def update(self, frame):
        """Update both data and performance plots."""
        # Update data plot
        self.line.set_data(list(self.times), list(self.values))
        
        plot_elements = [self.line]
        for name, scatter in self.scatter_plots.items():
            scatter.set_offsets(np.c_[list(self.anomalies[name]['x']),
                                      list(self.anomalies[name]['y'])])
            plot_elements.append(scatter)

        for name, times in self.detection_times.items():
            if not hasattr(self, f'time_line_{name}'):
                # Create line if it doesn't exist
                line, = self.ax2.plot([], [], label=name)
                setattr(self, f'time_line_{name}', line)
            
            line = getattr(self, f'time_line_{name}')
            line.set_data(list(range(len(self.times) - len(times), len(self.times))),
                         list(times))

        # Adjust plot limits
        if len(self.times) > 0:
            self.ax1.set_xlim(max(0, max(self.times) - 200), max(self.times) + 10)
            value_range = max(self.values) - min(self.values)
            self.ax1.set_ylim(min(self.values) - value_range * 0.1,
                              max(self.values) + value_range * 0.1)
        
            # Add limits for detection times plot
            self.ax2.set_xlim(self.ax1.get_xlim())
            if any(len(times) > 0 for times in self.detection_times.values()):
                all_times = [t for times in self.detection_times.values() for t in times]
                if all_times:  # Check if we have any times
                    self.ax2.set_ylim(0, max(all_times) * 1.1)
        
        self.ax2.legend()  
        return plot_elements

    # this method updates detection times
    def add_detection_time(self, detector_name, time_value):
        """Add a new detection time for a specific detector."""
        self.detection_times[detector_name].append(time_value)

class StreamProcessor:
    """Enhanced main processor with multiple detectors and performance tracking."""
    
    def __init__(self, config_path: str = None):
        self.config = ConfigManager.load_config(config_path)
        self.generator = DataStreamGenerator(self.config)
        self.detector_ensemble = DetectorEnsemble(self.config)
        self.visualizer = RealTimeVisualizer(self.config)
        self.running = True
        self.performance_metrics = PerformanceMetrics()
    
    def process_stream(self):
        """Main method to run the data stream processing and visualization."""
        self.running = True
        
    def start_stream(self):
        """Start generating and processing the data stream."""
        while self.running:
            value = self.generator.next_value()
            self.visualizer.times.append(self.generator.time)
            self.visualizer.values.append(value)

            anomalies = self.detector_ensemble.check_anomaly(value)
            for name, (is_anomalous, score) in anomalies.items():
                if is_anomalous:
                    self.visualizer.anomalies[name]['x'].append(self.generator.time)
                    self.visualizer.anomalies[name]['y'].append(value)

            time.sleep(self.config['visualization']['update_interval'] / 1000.0)

    def stop_stream(self):
        """Stop the data stream."""
        self.running = False
        plt.close(self.visualizer.fig)
        self._save_performance_metrics()
        

    def run_visualization(self):
        """Run the visualization in a separate thread."""
        ani = FuncAnimation(self.visualizer.fig, self.visualizer.update, interval=50)
        plt.show()
    def __init__(self, config_path: str = None):
        self.config = ConfigManager.load_config(config_path)
        self.generator = DataStreamGenerator(self.config)
        self.detector_ensemble = DetectorEnsemble(self.config)
        self.visualizer = RealTimeVisualizer(self.config)
        self.running = False
        self.performance_metrics = PerformanceMetrics()
    
    def process_stream(self):
        """Process the data stream with multiple detection algorithms."""
        current_time = 0
        
        while self.running:
            start_time = time.time()
            
            # Generate new value
            value = self.generator.next_value()
            
            # Detect anomalies using all detectors
            detection_results = {}
            for detector_name, detector in self.detector_ensemble.detectors.items():
                start_time = time.time()
                is_anomaly, score = detector.is_anomaly(value)
                detection_time = time.time() - start_time
            
            # Store  results and pass detection time to visualizer
                detection_results[detector_name] = (is_anomaly, score)
                self.visualizer.add_detection_time(detector_name, detection_time)

            # Update visualization data
            self.visualizer.times.append(current_time)
            self.visualizer.values.append(value)
            
             # Process detection results
            for detector_name, (is_anomaly, score) in detection_results.items():
                if is_anomaly:
                    self.visualizer.anomalies[detector_name]['x'].append(current_time)
                    self.visualizer.anomalies[detector_name]['y'].append(value)
                    logging.info(f"{detector_name} anomaly detected at {current_time}: "
                                 f"value={value:.2f}, score={score:.2f}")
            
            # Track performance
            processing_time = time.time() - start_time
            self.performance_metrics.add_metric(processing_time, 
                                                processing_time,
                                                len(self.visualizer.times))
            
            current_time += 1
            time.sleep(0.05)
    
    def start(self):
        """Start the streaming process with visualization."""
        self.running = True
        
        # Start processing thread
        process_thread = Thread(target=self.process_stream)
        process_thread.daemon = True
        process_thread.start()
        
        # Start visualization
        anim = FuncAnimation(self.visualizer.fig, 
                             self.visualizer.update,
                             interval=self.config['visualization']['update_interval'], 
                             blit=True)
        plt.show()
    
    def stop(self):
        """Stop the streaming process and save performance metrics."""
        self.running = False
        self._save_performance_metrics()
    
    def _save_performance_metrics(self):
        """Save performance metrics to JSON file."""
        metrics_summary = self.performance_metrics.get_summary()
        try:
            with open('performance_metrics.json', 'w') as f:
                json.dump(metrics_summary, f, indent=4)
            logging.info("Performance metrics saved successfully")
        except Exception as e:
            logging.error(f"Failed to save performance metrics: {e}")

def main():
    """Main function to run the enhanced anomaly detection system."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create a sample config file
    sample_config = {
        'data_generation': {
            'base_seasonal_period': 24,
            'secondary_seasonal_period': 168,
            'noise_level': 0.15,
            'anomaly_rate': 0.03,
            'trend_coefficient': 0.002
        },
        'detection': {
            'window_size': 150,
            'z_score_threshold': 3.5,
            'ewma_alpha': 0.15,
            'isolation_forest_samples': 256,
            'robust_zscore_threshold': 3.5
        },
        'visualization': {
            'max_points': 1000,
            'update_interval': 50
        }
    }
    
    # Save sample config
    with open('config.yaml', 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False)
    
    # Initialize and run the processor
    processor = StreamProcessor('config.yaml')
    
    try:
        processor.start()
    except KeyboardInterrupt:
        logging.info("Shutting down the system...")
        processor.stop()
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        processor.stop()

if __name__ == "__main__":
    main()
