import numpy as np

def compute_log_energy(signal, samples_per_window):
    """
    Returns a vector of the energy in each frame.
    A frame is, usually, 10 milliseconds worth of samples.
    Frames do not overlap.
    Thus the values returned in log_energy are the energy in the frames
    centered at 5ms, 15ms, 25 ms ...
    
    A typical call will be: en = compute_log_energy(signal, 80)
    
    Note that using the integral image risks overflow, so we convert to double.
    For a 10-minute file, at an 8K rate, there are only 5 million samples,
    and the max absolute sample value is about 20,000, and we square them,
    so the cumsum should always be under 10 to the 17th, so it should be safe.
    
    Args:
    signal (ndarray): The input signal.
    samples_per_window (int): Number of samples per analysis window.
    
    Returns:
    log_energy (ndarray): A vector of log energy values for each frame.
    """
    signal_array = np.array(signal, dtype=np.double)  # Convert input signal to NumPy array
    squared_signal = signal_array * signal_array
    integral_image = np.concatenate(([0], np.cumsum(squared_signal)))
    integral_image_by_frame = integral_image[::samples_per_window][1:]
    per_frame_energy = integral_image_by_frame[1:] - integral_image_by_frame[:-1]
    per_frame_energy = np.sqrt(per_frame_energy)
    
    # Replace zeros with a small positive value (1) to prevent log(0)
    zero_indices = np.where(per_frame_energy == 0)
    per_frame_energy[zero_indices] = 1.0
    
    log_energy = np.log(per_frame_energy)
    
    return log_energy


# Test cases:
# signal = Your input signal (e.g., output of the librosa.load)
# samples_per_window = Number of samples per analysis window
# log_energy = compute_log_energy(signal, samples_per_window)
# print(compute_log_energy([1,2,3], 1))
