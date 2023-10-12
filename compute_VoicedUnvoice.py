import numpy as np

def voiced_unvoiced_ir(log_energy, pitch, ms_per_window):
    # Compute voiced-unvoiced intensity ratio

    if len(log_energy) == len(pitch) + 1:
        pitch = [np.nan] + pitch

    is_speech = speaking_frames(log_energy)

    voiced_speech_vec = (~np.isnan(pitch) & is_speech)
    unvoiced_speech_vec = (np.isnan(pitch) & is_speech)

    non_voiced_energies_zeroed = voiced_speech_vec * log_energy
    non_unvoiced_energies_zeroed = unvoiced_speech_vec * log_energy

    v_frame_cum_sum = [0] + list(np.cumsum(non_voiced_energies_zeroed))
    u_frame_cum_sum = [0] + list(np.cumsum(non_unvoiced_energies_zeroed))

    v_frame_cum_count = [0] + list(np.cumsum(voiced_speech_vec))
    u_frame_cum_count = [0] + list(np.cumsum(unvoiced_speech_vec))

    frames_per_window = ms_per_window / 10
    frames_per_half_window = frames_per_window / 2

    v_frame_win_sum = np.zeros(len(pitch))
    u_frame_win_sum = np.zeros(len(pitch))
    v_frame_count_sum = np.zeros(len(pitch))
    u_frame_count_sum = np.zeros(len(pitch))

    for i in range(len(pitch)):
        w_start = int(i - frames_per_half_window)
        w_end = int(i + frames_per_half_window)

        # Prevent out-of-bounds
        if w_start < 0:
            w_start = 0
        if w_end >= len(pitch):
            w_end = len(pitch) - 1

        v_frame_win_sum[i] = v_frame_cum_sum[w_end] - v_frame_cum_sum[w_start]
        u_frame_win_sum[i] = u_frame_cum_sum[w_end] - u_frame_cum_sum[w_start]
        v_frame_count_sum[i] = v_frame_cum_count[w_end] - v_frame_cum_count[w_start]
        u_frame_count_sum[i] = u_frame_cum_count[w_end] - u_frame_cum_count[w_start]

    avg_voiced_intensity = np.zeros(len(pitch))
    avg_unvoiced_intensity = np.zeros(len(pitch))

    for i in range(len(pitch)):
        if v_frame_count_sum[i] != 0:
            avg_voiced_intensity[i] = v_frame_win_sum[i] / v_frame_count_sum[i]

        if u_frame_count_sum[i] != 0:
            avg_unvoiced_intensity[i] = u_frame_win_sum[i] / u_frame_count_sum[i]

    ratio = avg_voiced_intensity / avg_unvoiced_intensity

    # Exclude zeros, NaNs, and Infs
    average_of_valid = np.nanmean(ratio[~np.isinf(ratio) & (ratio > 0)])
    ratio = ratio - average_of_valid
    ratio[np.isnan(ratio)] = 0
    ratio[np.isinf(ratio)] = 0

    return ratio

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
    previous_low_center = np.nanmin(values)
    previous_high_center = np.nanmax(values)
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
        subset_average = np.nanmean(closer_samples)
    return subset_average

# 示例数据

e1 = [4, 5, 4, 5, 3, 5, 7, 4, 8, 9, 1, 8, 9, 9, 8, 7, 6, 9]
p1 = [2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7, 8, 7, 8, 7, 8, 9, 8, 9, 7]
ms_per_window2 = 200

e2 = [4, 0, 1, 1, 2, 1, 0, 1, 2, 7, 8, 7, 7, 7, 7]
p2 = [2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
ms_per_window3 = 200

result1 = voiced_unvoiced_ir(e1, p1, ms_per_window2)
result2 = voiced_unvoiced_ir(e2, p2, ms_per_window3)

print("Result 1:", result1)
print("Result 2:", result2)
