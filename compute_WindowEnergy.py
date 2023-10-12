import numpy as np

def window_energy(log_energy, ms_per_window):
    integral_image = np.concatenate([[0], np.cumsum(log_energy)])
    frames_per_window = int(ms_per_window / 10)  # Cast to integer
    window_sum = integral_image[frames_per_window:] - integral_image[:-frames_per_window]

    # Find silence and speech mean of track using k-means, then use them
    # to normalize for robustness against recording volume
    silence_mean, speech_mean = find_cluster_means(window_sum)
    difference = speech_mean - silence_mean

    if difference > 0:
        scaled_sum = (window_sum - silence_mean) / difference
    else:
        # Something's wrong; typically the file is mostly music or has a terribly low SNR.
        # So we just return something that at least has no NaNs, though it may or may not be useful.
        scaled_sum = (window_sum - (0.5 * silence_mean)) / silence_mean

    # Align so the first value is for the window centered at 10 ms (or 15ms if an odd number of frames)
    # Using zeros for padding is not ideal.
    head_frames_to_pad = int(frames_per_window / 2) - 1
    tail_frames_to_pad = int(frames_per_window / 2)
    win_energy = np.concatenate([np.zeros(head_frames_to_pad), scaled_sum, np.zeros(tail_frames_to_pad)])

    return win_energy

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
    # Returns the averages of all points which are closer to the near mean
    # than to the far mean
    
    # To save time, approximate by taking a sample of 2000 values.
    nsamples = 2000

    if len(values) < 2000:
        samples = values
    else:
        samples = values[::round(len(values) / nsamples)]

    near_mean = float(near_mean)
    far_mean = float(far_mean)

    closer_samples = [sample for sample in samples if abs(sample - near_mean) < abs(sample - far_mean)]

    if len(closer_samples) == 0:
        # If no closer samples are found, return a weighted average
        subset_average = 0.9 * near_mean + 0.1 * far_mean
    else:
        subset_average = np.mean(closer_samples)

    return subset_average

# Test cases:
# print(window_energy(np.array([0, 1, 1, 1, 2, 3, 3, 3, 1, 1, 1, 2]), 20))
print(window_energy(np.array([0, 1, 1, 1, 2, 3, 3, 3, 1, 1, 1, 2]), 30))

