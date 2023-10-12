import numpy as np

def speaking_frames(log_energy):
    # Returns a vector of 1s and 0s

    # Find silence and speech mean of track using k-means
    silence_mean, speech_mean = find_cluster_means(log_energy)

    # Set the speech/silence threshold closer to the silence mean
    # because the variance of silence is less than that of speech.
    # This is ad hoc; modeling with two Gaussians would probably be better
    threshold = (2 * silence_mean + speech_mean) / 3.0

    vec = log_energy > threshold
    return vec

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

print(speaking_frames([10, 20, 30, 40, 50, 60, 70, 80, 10, 20, 30, 40, 50, 60, 70]))