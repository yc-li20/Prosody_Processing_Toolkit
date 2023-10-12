import numpy as np

def compute_rate(log_energy, window_size_ms):
    frames = int(window_size_ms / 10)
    
    # Compute inter-frame deltas
    deltas = np.abs(np.diff(log_energy))
    cum_sum_deltas = np.concatenate(([0], np.cumsum(deltas)))
    
    # Adjust the indices to ensure compatible shapes
    window_liveliness = cum_sum_deltas[int(frames):] - cum_sum_deltas[:-int(frames)]
    
    # Normalize rate for robustness against recording volume
    silence_mean, speech_mean = find_cluster_means(log_energy)
    scaled_liveliness = (window_liveliness - silence_mean) / (speech_mean - silence_mean)
    
    head_frames_to_pad = int(np.floor(frames / 2)) - 1
    tail_frames_to_pad = int(np.ceil(frames / 2)) - 1
    scaled_liveliness = np.concatenate((np.zeros(head_frames_to_pad), 
                                        scaled_liveliness, 
                                        np.zeros(tail_frames_to_pad)))
    
    return scaled_liveliness

def find_cluster_means(values):
    max_iterations = 20
    previous_low_center = np.min(values)
    previous_high_center = np.max(values)
    convergence_threshold = (previous_high_center - previous_low_center) / 100

    for _ in range(max_iterations):
        high_center = average_of_near_values(values, previous_high_center, previous_low_center)
        low_center = average_of_near_values(values, previous_low_center, previous_high_center)

        # Check for convergence based on a small threshold
        if (abs(high_center - previous_high_center) < convergence_threshold and
            abs(low_center - previous_low_center) < convergence_threshold):
            return low_center, high_center

        previous_high_center = high_center
        previous_low_center = low_center

    # If max iterations are reached without convergence, issue a warning
    print('findClusterMeans exceeded maxIterations without converging')
    print(previous_high_center)
    print(previous_low_center)
    return previous_low_center, previous_high_center

def average_of_near_values(values, near_mean, far_mean):
    nsamples = 2000

    if len(values) < 2000:
        samples = values
    else:
        samples = values[::round(len(values) / nsamples)]

    near_mean = float(near_mean)
    far_mean = float(far_mean)

    closer_samples = [sample for sample in samples if abs(sample - near_mean) < abs(sample - far_mean)]

    if len(closer_samples) == 0:
        subset_average = 0.9 * near_mean + 0.1 * far_mean
    else:
        subset_average = np.mean(closer_samples)

    return subset_average

# Test cases:
# log_energy = Your input log energy values
# window_size_ms = Size of the analysis window in milliseconds
# scaled_liveliness = compute_rate(log_energy, window_size_ms)
# print(compute_rate([1.0, 2.0, 3.0, 2.5, 2.7, 3.5, 2.0, 1.8], 20))
