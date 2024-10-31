import numpy as np


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