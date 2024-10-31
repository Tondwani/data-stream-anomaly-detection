Configuration
Configuration is managed in config/config.yaml. Parameters include:

Data Generation:
base_seasonal_period: Primary seasonal period (e.g., daily).
secondary_seasonal_period: Secondary seasonal pattern (e.g., weekly).
noise_level: Random noise level in data.
anomaly_rate: Frequency of anomalies.
Detection:
window_size: Number of past observations to consider.
z_score_threshold, robust_zscore_threshold, and ewma_alpha: Thresholds for each detection method.
Visualization:
max_points: Maximum number of data points to display.
update_interval: Refresh interval for visualizer.