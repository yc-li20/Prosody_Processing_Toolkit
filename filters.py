import numpy as np

def rectangular_filter(window_duration_ms):
    duration_frames = int(np.floor(window_duration_ms / 10))
    filter_values= np.ones(duration_frames) / duration_frames
    return filter_values
    
def triangle_filter(window_duration_ms):
    duration_frames = int(np.floor(window_duration_ms / 10))
    center = int(np.floor(duration_frames / 2))
    filter_values = [center - abs(i - center) for i in range(1, duration_frames + 1)]
    filter_values = np.array(filter_values) / np.sum(filter_values)  # normalize to sum to one
    return filter_values