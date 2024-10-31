import json
import logging
import os
import sys
import time
from threading import Thread

import matplotlib.pyplot as plt
import yaml
from matplotlib.animation import FuncAnimation

sys.path.insert(0, "C:/Users/Craig/Desktop/MyGitHub Projects/Data Stream Anomaly Detection/src")

from anomaly_detection.detectors import PerformanceMetrics
from data_generation.data_stream_generation import DataStreamGenerator
from ensemble.ensemble_detectors import DetectorEnsemble
from visualization.visualizer import RealTimeVisualizer


class ConfigManager:
    """Manages configuration loading and validation."""
    
    DEFAULT_CONFIG = {
        'data_generation': {
            'base_seasonal_period': 24,
            'secondary_seasonal_period': 168,
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

class StreamProcessor:
    """Enhanced main processor with multiple detectors and performance tracking."""
    
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
            
                # Store results and pass detection time to visualizer
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