This document provides a reference for the core classes and methods:

ConfigManager: Manages loading and validation of the configuration file.
load_config(config_path: str): Loads config from a specified path.
DataStreamGenerator: Generates data with seasonal patterns and anomalies.
next_value() -> float: Generates the next value in the data stream.
AnomalyDetectorBase: Base class for anomaly detection algorithms.
is_anomaly(value: float) -> Tuple[bool, float]: Determines if a value is an anomaly.
DetectorEnsemble: Combines multiple detectors.
check_anomaly(value: float) -> Dict[str, Tuple[bool, float]]: Aggregates anomaly checks from all detectors.
RealTimeVisualizer: Displays data and detected anomalies.
setup_plots(): Initializes plot configurations for real-time data visualization.