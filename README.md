# Efficient  Real-Time Data Stream Anomaly Detection

This project is part of an assessment for Cobblestone Energy's Graduate Software Engineer role. It detects anomalies in a continuous data stream using Python. A robust Python application for real-time anomaly detection in data streams using multiple detection algorithms and interactive visualization. The system implements ensemble-based anomaly detection with support for multiple seasonal patterns and various types of anomalies.

## Project Overview
The Python script simulates a real-time data stream and detects anomalies using a Z-score algorithm. It visualizes the data stream and flags anomalies in real-time.

## Features

- **Multiple Anomaly Detection Algorithms**:
  - Z-Score Detection
  - Exponentially Weighted Moving Average (EWMA)
  - Robust Z-Score Detection using Median and MAD
  
- **Advanced Data Generation**:
  - Multiple seasonal patterns (daily, weekly, monthly)
  - Configurable trend and noise levels
  - Various anomaly types (spikes, level shifts, trend changes)

- **Real-time Visualization**:
  - Interactive data stream plotting
  - Anomaly highlighting for each detector
  - Performance metrics monitoring
  - Detection time tracking

- **Performance Monitoring**:
  - Processing time tracking
  - Detection time measurements
  - Memory usage monitoring
  - Metrics export functionality

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tondwani/data-stream-anomaly-detection.git
cd data-stream-anomaly-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure the system by modifying `config.yaml` (a default configuration will be created on first run):
```yaml
data_generation:
  base_seasonal_period: 24
  secondary_seasonal_period: 168
  noise_level: 0.15
  anomaly_rate: 0.03
  trend_coefficient: 0.002
```

2. Run the application:
```bash
python src/main.py
```

3. To stop the application, press Ctrl+C in the terminal. The system will save performance metrics before shutting down.

## Project Structure

```
Efficiency data-stream-anomaly-detection/
├── config/
|   ├── config_schema.json
|
├── Docs/
|   ├──api-reference.md
|   ├──configuration.md
|   ├──getting-started.md
|   ├──overview.md
|
├── images/
├──py src/
│   ├── main.
│   ├── data_generation/
│   │   └── data_stream_generator.py
│   ├── anomaly_detection/
│   │   └── detectors.py
│   ├── ensemble/
│   │   └── ensemble_detectors.py
│   └── visualization/
│       └── visualizer.py
├── .gitignore
├── config.yaml
├── requirements.txt
└── README.md
```

## Configuration Options

The system can be configured through `config.yaml` with the following sections:

### Data Generation
- `base_seasonal_period`: Primary seasonal pattern period (default: 24)
- `secondary_seasonal_period`: Secondary seasonal pattern period (default: 168)
- `noise_level`: Amount of random noise in the data (default: 0.15)
- `anomaly_rate`: Probability of generating anomalies (default: 0.03)
- `trend_coefficient`: Strength of the underlying trend (default: 0.002)

### Detection
- `window_size`: Number of observations used for detection (default: 150)
- `z_score_threshold`: Threshold for Z-score detection (default: 3.5)
- `ewma_alpha`: EWMA smoothing factor (default: 0.15)
- `robust_zscore_threshold`: Threshold for robust Z-score detection (default: 3.5)

### Visualization
- `max_points`: Maximum number of points to display (default: 1000)
- `update_interval`: Visualization update interval in milliseconds (default: 50)

## Performance Metrics

The system automatically saves performance metrics to `config_schema.json`, including:
- Average and maximum processing times
- Average and maximum detection times
- Memory usage statistics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
