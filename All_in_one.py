# mid-level prosodic feature extraction and related functions
# partly adapted from Nigel's midlevel matlab toolkit: https://github.com/nigelgward/midlevel/tree/master
# Yuanchao Li, CSTR, University of Edinburgh

import numpy as np
from scipy.signal import spectrogram, lfilter, convolve, linregress

# --- calculate Smoothed Cepstral Peak Prominence (CPPS) ---
def compute_CPPS(signal, sample_rate):
    midlevel_frame_width_ms = 10
    midlevel_frame_width_s = midlevel_frame_width_ms / 1000
    signal_duration_s = len(signal) / sample_rate
    signal_duration_ms = signal_duration_s * 1000
    expected_CPPS_midlevel_len = int(np.floor(signal_duration_ms / midlevel_frame_width_ms))

    # Window analysis settings
    win_len_s = 0.048
    win_step_s = midlevel_frame_width_s
    win_len = int(round(win_len_s * sample_rate))
    win_step = int(round(win_step_s * sample_rate))
    win_overlap = win_len - win_step

    # Quefrency range
    quef_bot = int(round(sample_rate / 300))
    quef_top = int(round(sample_rate / 60))
    quefs = np.arange(quef_bot, quef_top + 1)

    # Pre-emphasis from 50 Hz
    alpha = np.exp(-2 * np.pi * 50 / sample_rate)
    signal = lfilter([1, -alpha], 1, signal)

    # Compute spectrogram
    f, t, spec = spectrogram(signal, fs=sample_rate, nperseg=win_len, noverlap=win_overlap)
    spec_power = np.abs(spec) ** 2  # Calculate power from the complex values
    spec_log = 10 * np.log10(spec_power)

    # Compute cepstrum
    ceps_log = 10 * np.log10(np.abs(np.fft.fft(spec_log, axis=0)) ** 2)

    # Do time- and quefrency-smoothing
    smooth_filt_samples = np.ones(2) / 2
    smooth_filt_quef = np.ones(10) / 10
    ceps_log_smoothed = lfilter(smooth_filt_samples, 1, lfilter(smooth_filt_quef, 1, ceps_log, axis=0), axis=0)

    # Find cepstral peaks in the quefrency range
    ceps_log_smoothed = ceps_log_smoothed[quefs, :]
    peak_quef = np.argmax(ceps_log_smoothed, axis=0)

    # Get the regression line and calculate its distance from the peak
    n_wins = ceps_log_smoothed.shape[1]
    ceps_norm = np.zeros(n_wins)

    for n in range(n_wins):
        p = np.polyfit(quefs, ceps_log_smoothed[:, n], 1)
        ceps_norm[n] = np.polyval(p, quefs[peak_quef[n]])

    cpps = np.max(ceps_log_smoothed, axis=0) - ceps_norm

    # Pad the CPPS vector and calculate means in 10-ms window
    pad_size = expected_CPPS_midlevel_len - len(cpps)
    prepad_size = pad_size // 2
    postpad_size = pad_size - prepad_size
    cpps_padded = np.concatenate((np.full(prepad_size, np.nan), cpps, np.full(postpad_size, np.nan)))
    CPPS_midlevel = cpps_padded
    CPPS_midlevel[np.isnan(CPPS_midlevel)] = np.nanmedian(CPPS_midlevel)

    return CPPS_midlevel

# Test case
# sample_rate = 100  # Sample rate in Hz
# signal = np.random.rand(100)  # Example signal
# print(compute_CPPS(signal, sample_rate))


# --- calculate creakiness ---
def compute_creakiness(pitch, window_size_ms):
    # Number of frames per window
    ms_per_window = 10
    frames_per_window = int(window_size_ms / ms_per_window)

    # Calculate frequency ratios between adjacent frames
    ratios = np.array(pitch[1:]) / np.array(pitch[:-1])

    # Check for octave jumps and other frame-to-frame frequency jumps
    octave_up = (ratios > 1.90) & (ratios < 2.10)
    octave_down = (ratios > 0.475) & (ratios < 0.525)
    small_up = (ratios > 1.05) & (ratios < 1.25)
    small_down = (ratios < 0.95) & (ratios > 0.80)

    # Compute creakiness for each frame
    creakiness = octave_up + octave_down + small_up + small_down

    # Calculate the integral image
    integral_image = np.concatenate(([0], np.cumsum(creakiness)))
    creakiness_per_window = integral_image[frames_per_window:].astype(float) - integral_image[:-frames_per_window].astype(float)

    # Pad in front and at the end
    head_frames_to_pad = int(np.ceil((frames_per_window - 1) / 2))
    tail_frames_to_pad = int(np.ceil(frames_per_window / 2))
    creak_array = np.concatenate((np.zeros(head_frames_to_pad), creakiness_per_window, np.zeros(tail_frames_to_pad)))
    creak_values = creak_array / frames_per_window

    return creak_values

# Test case
# pitch = Your pitch data, e.g., [1, 1, 1, 1.1, 0.8, 1.0, 0.9, 1, 1, 1, 1.01, 1.02, 1]
# window_size_ms = Your window size in milliseconds
# creak_values = compute_creakiness(pitch, window_size_ms)
# print(compute_creakiness(pitch, 10))


# --- calculate energy stability ---
def compute_energy_stability(energy, window_size):
    range_count = []
    ms_per_window = 10
    frames_per_window = int(window_size / ms_per_window)
    relevant_span = 200  # Temporary value
    frames_per_half_span = int((relevant_span / 2) / frames_per_window)

    for i in range(len(energy)):
        # Calculate the offset for a 500 ms window
        start_neighbors = i - frames_per_half_span
        end_neighbors = i + frames_per_half_span
        # Ensure the bounds are within the data
        if start_neighbors < 0:
            start_neighbors = 0
        if end_neighbors >= len(energy):
            end_neighbors = len(energy)

        # Extract neighbors with the energy point in the center
        neighbors = energy[start_neighbors:end_neighbors]
        # Calculate evidence based on ratios to the center energy
        ratios = [neighbor / energy[i] for neighbor in neighbors]
        # Count points within the specified pitch range based on ratio difference
        count = sum(0.90 < ratio < 1.10 for ratio in ratios)
        range_count.append(count)

    # Compute an integral image for efficiency
    integral_image = np.concatenate(([0], np.cumsum(range_count)))
    window_values = integral_image[frames_per_window:].astype(float) - integral_image[:-frames_per_window].astype(float)

    padding_needed = frames_per_window - 1
    front_padding = np.zeros(int(padding_needed / 2))
    tail_padding = np.zeros(int(np.ceil(padding_needed / 2)))
    energy_range = np.concatenate((front_padding, window_values, tail_padding))
    energy_stability = energy_range / frames_per_window

    return energy_stability

# Test case
# energy = Your energy data
# window_size = Your window size in milliseconds, e.g., 100
# energy_range = compute_energy_stability(energy, window_size)
# print(compute_energy_stability([100, 100, 120, 300, 400, 500, 600], 10))


# ---calculate pitch range ---
def compute_pitch_range(pitch, window_size, range_type):
    #  f = flat (within 0.99-1.0 difference)
    #  n = narrow (within 0.98-1.02 difference)
    #  w = wide (within 0.70-0.90 difference)

    range_count = []
    ms_per_window = 10
    frames_per_window = int(window_size / ms_per_window)
    relevant_span = 1000
    frames_per_half_span = int(np.floor((relevant_span / 2) / frames_per_window))

    for i in range(len(pitch)):
        # Calculate the offset for a 500 ms window
        start_neighbors = i - frames_per_half_span
        end_neighbors = i + frames_per_half_span
        # Ensure the bounds are within the data
        if start_neighbors < 0:
            start_neighbors = 0
        if end_neighbors >= len(pitch):
            end_neighbors = len(pitch)

        # Extract neighbors with the pitch point in the center
        # Here, we could take every other point to save time, with probably
        # no performance penalty, as it doesn't change much over just 10ms
        neighbors = pitch[start_neighbors:end_neighbors]
        # Calculate evidence based on ratios to the center pitch
        ratios = [neighbor / pitch[i] for neighbor in neighbors]

        # Based on ratio difference to center, count points with evidence
        # for the specified pitch range
        if range_type == 'f':
            # Doesn't seem to be usefully different from narrow
            count = sum(0.99 <= ratio <= 1.01 for ratio in ratios)
            range_count.append(count)
        elif range_type == 'n':
            count = sum(0.98 <= ratio <= 1.02 for ratio in ratios)
            range_count.append(count)
        elif range_type == 'w':
            # If the difference is <0.70 or >1.3, it's most likely a spurious pitch point
            count = sum((0.70 < ratio < 0.90) or (1.1 < ratio < 1.3) for ratio in ratios)
            range_count.append(count)


    # Same old trick of integral image
    integral_image = np.concatenate(([0], np.cumsum(range_count)))
    window_values = integral_image[frames_per_window:].astype(float) - integral_image[:-frames_per_window].astype(float)

    padding_needed = frames_per_window - 1
    front_padding = np.zeros(int(np.floor(padding_needed / 2)))
    tail_padding = np.zeros(int(np.ceil(padding_needed / 2)))
    pitch_range = np.concatenate((front_padding, window_values, tail_padding))
    pitch_range = pitch_range / frames_per_window

    return pitch_range

# Test case
# pitch = Your pitch data
# window_size = Your window size in milliseconds, e.g., 100
# range_type = 'f' for flat, 'n' for narrow, 'w' for wide
# pitch_range = compute_pitch_range(pitch, window_size, range_type)
# print(compute_pitch_range([100, 200, 300, 100, 100], 10, 'f'))


# --- percentilize pitch ---
def percentilize_pitch(pitch_points, max_pitch):
    # Create a percentile vector to represent pitch values
    # Instead of using specific Hz values, it maps them to percentiles in the overall distribution.
    # This non-linearly scales the input while preserving NaNs.
    # It accepts either a row vector or a column vector and returns a column vector.

    # The input pitch_points typically range from 50 to around 515.
    # any points above max_pitch are mapped to NaN.

    rounded = np.asarray(pitch_points, dtype=np.float64).round()

    # Initialize a histogram of the distribution
    counts = np.zeros(max_pitch, dtype=np.int64)

    # Build the histogram
    for pitch in rounded:
        if 1 <= pitch <= max_pitch:
            # It's within the range and not a NaN
            counts[int(pitch) - 1] += 1

    # Compute the cumulative sum of pitches that are less than or equal to each specified Hz value
    cumulative_sum = np.cumsum(counts)

    # Compute the fraction of all pitches that are less than or equal to each specified Hz value
    mapping = cumulative_sum / cumulative_sum[max_pitch - 1]

    percentiles = np.zeros(len(rounded))

    # Map each pitch to its percentile value
    for i in range(len(rounded)):
        pitch = rounded[i]
        if 1 <= pitch <= max_pitch:
            percentiles[i] = mapping[int(pitch) - 1]
        else:
            percentiles[i] = np.nan
    return percentiles


# Test case
# pitch_points = Your pitch data (e.g., [100, 200, 300, 100, 100, 100, 300, 100, 500, 140, 200, 300, NaN, 500, 600, 300, 500, 700, 100])
# max_pitch = Maximum pitch value (e.g., 500)
# percentiles = percentilize_pitch(pitch_points, max_pitch)
# print(percentile)


# --- compute log energy ---
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


# Test case
# signal = Your input signal (e.g., output of the librosa.load)
# samples_per_window = Number of samples per analysis window
# log_energy = compute_log_energy(signal, samples_per_window)
# print(compute_log_energy([1,2,3], 1))


# --- compute pitch in band ---
def compute_pitch_in_band(percentiles, band_flag, window_size_ms):
    """
    Compute evidence for the pitch being strongly in the specified band.
    
    Args:
    percentiles (ndarray): A vector of pitch percentiles.
    band_flag (str): Band flag indicating the pitch range ('h' for high, 'l' for low, 'th' for truly high, 'tl' for truly low).
    window_size_ms (int): Size of the analysis window in milliseconds.
    
    Returns:
    band_values (ndarray): A vector of band values.
    """
    percentiles = np.array(percentiles, dtype=float)  # Convert input to NumPy array with float dtype
    
    # Initialize evidence vector
    evidence_vector = None
    
    if band_flag == 'h':
        # For computing evidence for high pitch, NaNs contribute nothing, same as 0s
        percentiles[np.isnan(percentiles)] = 0.00
        evidence_vector = percentiles
    elif band_flag == 'l':
        # For computing evidence for low pitch, NaNs contribute nothing, same as 1s
        percentiles[np.isnan(percentiles)] = 1.00
        evidence_vector = 1 - percentiles  # The lower the pitch value, the more evidence
    elif band_flag == 'th':
        # 50th percentile counts a tiny bit "truly high", below 50th percentile, not at all
        percentiles[np.isnan(percentiles)] = 0.00
        percentiles[percentiles < 0.50] = 0.50
        evidence_vector = percentiles - 0.50
    elif band_flag == 'tl':
        # 49th percentile counts a tiny bit "truly low", above 50th percentile, not at all
        percentiles[np.isnan(percentiles)] = 1.00
        percentiles[percentiles > 0.50] = 0.50
        evidence_vector = 0.50 - percentiles
    else:
        print('Sorry, unknown flag:', band_flag)
    
    integral_image = np.concatenate(([0], np.cumsum(evidence_vector)))
    frames_per_window = int(window_size_ms / 10)
    window_values = integral_image[frames_per_window:].copy() - integral_image[:-frames_per_window].copy()
    
    # Add more padding to the front.
    # If frames_per_window is even, this means the first value will be at 15ms,
    # otherwise, it will be at 10ms.
    padding_needed = frames_per_window - 1
    front_padding = np.zeros(int(padding_needed / 2))
    tail_padding = np.zeros(int(np.ceil(padding_needed / 2)))
    band_values = np.concatenate((front_padding, window_values, tail_padding))
    
    # now normalize, just so that when we plot them, the ones with longer windows are not hugely higher than the rest
    band_values = band_values / frames_per_window
    
    return band_values

# Test case
# percentiles = Your input percentiles vector
# band_flag = 'h', 'l', 'th', or 'tl' based on the desired band
# window_size_ms = Size of the analysis window in milliseconds
# band_values = compute_pitch_in_band(percentiles, band_flag, window_size_ms)
# print(compute_pitch_in_band([1,1,1,0.9,'NaN',0.8,0.9,1.0,0.95,0.95],'h', 90))


# --- compute speaking rate ---
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
    rate = np.concatenate((np.zeros(head_frames_to_pad), 
                                        scaled_liveliness, 
                                        np.zeros(tail_frames_to_pad)))
    
    return rate

# Test case
# log_energy = Your input log energy values
# window_size_ms = Size of the analysis window in milliseconds
# rate = compute_rate(log_energy, window_size_ms)
# print(compute_rate([1.0, 2.0, 3.0, 2.5, 2.7, 3.5, 2.0, 1.8], 20))


# --- find cluster means ---
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

# Test case
# values = np.array([1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 4, 6, 7, 6, 7, 6, 7, 6, 7, 1, 9, 0, 6, 6, 3])
# low_center, high_center = find_cluster_means(values)
# print("Low Center:", low_center)
# print("High Center:", high_center)


# --- estimate pitch-peak delay ---
def disalignment(epeaky, ppeaky):
    # Find local maxima in epeaky, representing energy peaks
    local_max_epeak = find_local_max(epeaky, 120)
    
    # Calculate the expected product of local maxima and ppeaky
    expected_product = local_max_epeak * ppeaky
    
    # Calculate the actual product of epeaky and ppeaky
    actual_product = epeaky * ppeaky
    
    # Compute the disalignment estimate
    disalignment = (expected_product - actual_product) * ppeaky
    
    return disalignment


# ---find local maximum value---
def find_local_max(vector, width_ms):
    # Calculate half the width in frames
    half_width_frames = int((width_ms / 2) / 10)
    mx = np.zeros(len(vector))
    
    # Iterate through the vector
    for e in range(len(vector)):
        start_frame = max(0, e - half_width_frames)
        end_frame = min(e + half_width_frames, len(vector))
        
        # Find the maximum value within the defined window
        mx[e] = max(vector[start_frame:end_frame])
    
    return mx

# Test case
# print(find_local_max([1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 2, 4, 5, 7, 1, 1, 1, 1, 2], 60))
# It should return the local maxima in the given list.


# ---compute energy peakness ---
def epeakness(vec):
    iSFW = 6  # in-syllable filter width, in frames
    iFFW = 15  # in-foot filter width, in frames

    vec = np.array(vec)

    # Height normalization
    height = np.sqrt((vec - np.min(vec)) / (np.max(vec) - np.min(vec)))

    # Convolve with Laplacian of Gaussian filters
    inSyllablePeakness = myconv(vec, laplacian_of_gaussian(iSFW), iSFW * 2.5)
    inFootPeakness = myconv(vec, laplacian_of_gaussian(iFFW), iFFW * 2.5)

    # print(inSyllablePeakness, inFootPeakness)

    # Compute peakness as the product of the above values and height
    peakness = inSyllablePeakness * inFootPeakness * height

    # Remove negative values (we don't consider troughs when aligning)
    epeakness[epeakness < 0] = 0
    return epeakness

# ---compute pitch peakness ---
def ppeakness(pitch_ptile):
    ssFW = 10  # stressed-syllable filter width; could be 12

    # Identify valid pitch values and compute local pitch amount
    pitch_ptile = np.array(pitch_ptile)
    valid_pitch = pitch_ptile > 0
    local_pitch_amount = myconv(1.0 * valid_pitch, triangle_filter(160), 10)

    # Replace NaN values with 0
    pitch_ptile[np.isnan(pitch_ptile)] = 0

    # Convolve with Laplacian of Gaussian filter
    local_peakness = myconv(pitch_ptile, laplacian_of_gaussian(ssFW), 2.5 * ssFW)

    # Compute peakness as the product of the above values and pitch_ptile
    ppeakness = local_peakness * local_pitch_amount * pitch_ptile

    # Remove negative values (don't care about troughs)
    ppeakness[ppeakness < 0] = 0
    return ppeakness

# Test case
# print(epeakness([5, 2, 3, 8, 2, 5, ...])) usually should be much longer than iFFW * 2.5
# print(ppeakness([5, 2, 3, 8, 2, 5, ...])) should be much longer than 2.5 * ssFW


# --- compute spectral tilt ---
def compute_spectral_tilt(signal, samples_per_second):
    # Build array with the target frequency values starting at 50 Hz, up to
    # 4000 Hz, every third of an octave
    signal = np.array(signal)
    start_freq = 50
    end_freq = 4000
    intervals_per_octave = 3
    octave_range = np.log2(end_freq / start_freq)
    total_intervals = int(octave_range * intervals_per_octave)
    freq_values = [50]

    for i in range(1, total_intervals + 1):
        freq_value = start_freq * 2 ** (i / intervals_per_octave)
        freq_values.append(freq_value)

    freq_values.append(4000)

    window_size = int(0.025 * samples_per_second)  # 15ms windows
    overlap = int(0.015 * samples_per_second)  # To get a value for every 10 ms an overlap of 15 is needed
    frequencies, times, Sxx = spectrogram(signal, fs=samples_per_second, window='hann', nperseg=window_size, noverlap=overlap)

    # Build empty matrix for the amplitude values squaring the values
    # obtained from calling spectrogram
    signal_length = len(signal) / samples_per_second
    nframes = int(np.floor(signal_length / 0.01))  # number of 10ms windows (frames)
    nfrequencies = len(freq_values)
    amplitudes = np.abs(Sxx) ** 2  # calculate Power Spectral Density (PSD)
    A = np.zeros((nfrequencies, nframes))
    freq_range = 10  # define the frequency range in Hz

# Fill matrix with the average PSD at each target frequency every 10ms
    for i in range(nfrequencies):
        freq = freq_values[i]
        for j in range(amplitudes.shape[1]):
            freq_idx = (frequencies >= (freq - freq_range)) & (frequencies <= (freq + freq_range))
            
            # Check if there are enough data points to calculate the mean
            if np.sum(freq_idx) > 0:
                A[i, j] = np.mean(amplitudes[freq_idx, j])
            else:
                A[i, j] = 0.0  # Set to 0 if no data points are available

    print("Calculating spectral tilt every 10ms...")

    # Run linear regression in every column (every 10ms window)
    tilts = np.zeros(nframes)
    for j in range(nframes):
        slope, _ = linregress(freq_values, A[:, j])[:2]
        tilts[j] = slope
    return tilts

# Test case
# print(compute_spectral_tilt([300,200,300,200,100,200,300,400,50,600,20,300,400], 600))

# ---compute speaking frames ---
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

# Test case
# print(speaking_frames([10, 20, 30, 40, 50, 60, 70, 80, 10, 20, 30, 40, 50, 60, 70]))


# --- compute voiced-unvoiced intensity ratio ---
def voiced_unvoiced_ir(log_energy, pitch, ms_per_window):

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

# Test case
# e1 = [4, 5, 4, 5, 3, 5, 7, 4, 8, 9, 1, 8, 9, 9, 8, 7, 6, 9]
# p1 = [2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7, 8, 7, 8, 7, 8, 9, 8, 9, 7]
# ms_per_window2 = 200
# e2 = [4, 0, 1, 1, 2, 1, 0, 1, 2, 7, 8, 7, 7, 7, 7]
# p2 = [2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
# ms_per_window3 = 200
# result1 = voiced_unvoiced_ir(e1, p1, ms_per_window2)
# result2 = voiced_unvoiced_ir(e2, p2, ms_per_window3)
# print("Result 1:", result1)
# print("Result 2:", result2)


# --- compute windowed value ---
def windowize(frame_features, ms_per_window):
    # inputs:
    #   frame_features: features over every 10 millisecond frame,
    #     centered at 5ms, 15ms etc.
    #     A row vector.
    #   ms_per_window: duration of window over which to compute windowed values
    # output:
    #   summed values over windows of the designated size,
    #     centered at 10ms, 20ms, etc.
    #     (the centering is off, at 15ms, etc, if ms_per_window is 30ms, 50ms etc)
    #     but we're not doing syllable-level prosody, so it doesn't matter.
    #   values are zero if either end of the window would go outside
    #     what we have data for.

    integral_image = np.concatenate([[0], np.cumsum(frame_features)])
    frames_per_window = int(ms_per_window / 10)  # Cast to integer
    window_sum = integral_image[frames_per_window:] - integral_image[:-frames_per_window]

    # align so the first value is for the window centered at 10 ms
    head_frames_to_pad = int(frames_per_window / 2) - 1
    tail_frames_to_pad = int(frames_per_window / 2)
    window_values = np.concatenate([np.zeros(head_frames_to_pad), window_sum, np.zeros(tail_frames_to_pad)])
    return window_values

# Test case
# print(windowize(np.array([0, 1, 1, 1, 2, 3, 3, 3, 1, 1, 1, 2]), 20))


# --- compute window energy ---
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

# Test case
# print(window_energy(np.array([0, 1, 1, 1, 2, 3, 3, 3, 1, 1, 1, 2]), 20))
# print(window_energy(np.array([0, 1, 1, 1, 2, 3, 3, 3, 1, 1, 1, 2]), 30))


# --- related functions ---
def myconv(vector, kernel, filterHalfWidth):
    result = convolve(vector, kernel, mode='same')
    trimWidth = int(np.floor(filterHalfWidth))
    result[:trimWidth] = 0
    result[-trimWidth:] = 0
    return result

def laplacian_of_gaussian(sigma):
    # Length calculation
    length = int(sigma * 5)

    sigmaSquare = sigma * sigma
    sigmaFourth = sigmaSquare * sigmaSquare

    vec = np.zeros(length)
    center = length // 2
    for i in range(length):
        x = i - center
        y = ((x * x) / sigmaFourth - 1 / sigmaSquare) * np.exp((-x * x) / (2 * sigmaSquare))
        vec[i] = -y
    return vec

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
