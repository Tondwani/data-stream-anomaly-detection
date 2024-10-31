from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


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
        
        self.detection_times = {
            'zscore': deque(maxlen=self.max_points),
            'ewma': deque(maxlen=self.max_points),
            'robust_zscore': deque(maxlen=self.max_points)
        }

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
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
        self.line.set_data(list(self.times), list(self.values))
        
        plot_elements = [self.line]
        for name, scatter in self.scatter_plots.items():
            scatter.set_offsets(np.c_[list(self.anomalies[name]['x']),
                                      list(self.anomalies[name]['y'])])
            plot_elements.append(scatter)

        for name, times in self.detection_times.items():
            if not hasattr(self, f'time_line_{name}'):
                line, = self.ax2.plot([], [], label=name)
                setattr(self, f'time_line_{name}', line)
            
            line = getattr(self, f'time_line_{name}')
            line.set_data(list(range(len(self.times) - len(times), len(self.times))),
                         list(times))

        if len(self.times) > 0:
            self.ax1.set_xlim(max(0, max(self.times) - 200), max(self.times) + 10)
            value_range = max(self.values) - min(self.values)
            self.ax1.set_ylim(min(self.values) - value_range * 0.1,
                              max(self.values) + value_range * 0.1)
            
            self.ax2.set_xlim(self.ax1.get_xlim())
            if any(len(times) > 0 for times in self.detection_times.values()):
                all_times = [t for times in self.detection_times.values() for t in times]
                if all_times:
                    self.ax2.set_ylim(0, max(all_times) * 1.1)
        
        self.ax2.legend()  
        return plot_elements

    def add_detection_time(self, detector_name, time_value):
        """Add a new detection time for a specific detector."""
        self.detection_times[detector_name].append(time_value)