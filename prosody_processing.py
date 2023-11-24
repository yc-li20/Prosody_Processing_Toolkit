"""
np.array and individual file processing

Authors
 * Yuanchao Li 2023
"""


import torch
import librosa
import numpy as np
from scipy.stats import linregress
from scipy.signal import spectrogram, lfilter


def compute_pitch_energy(signal, sample_rate, win_length):
    """Compute pitch and energy of a batch of waveforms.

    Arguments
    ---------
    signal : list or array-like
        The waveforms used for computing pitch and energy.
    sample_rate : int
        The sample rate of the audio signal.
    win_length : int
        Frame length to process.

    Returns
    -------
    pitch : np.ndarray
        The f0 envelope of the input signal.
    energy : np.ndarray
        The energy envelope of the input signal.
    log_energy : np.ndarray
        The log energy of the input signal.

    Example
    -------
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> sample_rate = 16000
    >>> compute_pitch_energy(signal, sample_rate, win_length)
    (array([290.25, 287.69, ...]), array([90.25, 32.58, ...]), array([90.25, 32.58, ...]))  # Example output, actual values may vary
    """

    # Compute the short-time Fourier transform
    signal = np.array(signal)
    stft = librosa.stft(signal)

    # Calculate the magnitude spectrum
    magnitude = np.abs(stft)
    
    # Compute the energy envelope
    energy = np.sum(magnitude, axis=0)

    log_energy = np.log(energy + 1e-12) # plus a small positive value to prevent log(0)

    # Compute the f0 (fundamental frequency) using the harmonic product spectrum (HPS) method
    pitch, voiced_flag, voiced_probs = librosa.pyin(signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=16000)

    return energy, log_energy, pitch


def compute_CPPS(signal, sample_rate, win_length):
    """Compute Smoothed Cepstral Peak Prominence (CPPS) of an input signal.

    Arguments
    ---------
    signal : list or array-like
    sample_rate : int
        Sample rate of the input audio signal (e.g, 16000).
    win_length : int
        Frame length to process.

    Returns
    -------
    The Smoothed Cepstral Peak Prominence of the input signal.

    Example
    -------
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')  # make sure the sample is 1-channel
    >>> sample_rate = 16000
    >>> win_length = 10
    >>> compute_CPPS(signal, sample_rate, win_length)
    array([[2.46409389, 3.09066041, ..., 6.59823178, 2.385688]])
    """

    win_length_s = win_length / 1000
    signal_duration_s = len(signal) / sample_rate
    signal_duration_ms = signal_duration_s * 1000
    expected_CPPS_midlevel_len = int(np.floor(signal_duration_ms / win_length))

    # Window analysis settings
    win_len_s = 0.048
    win_step_s = win_length_s
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
    CPPS_value = cpps_padded
    CPPS_value[np.isnan(CPPS_value)] = np.nanmedian(CPPS_value)

    return CPPS_value


def compute_creakiness(pitch, win_length):
    """Compute creakiness of the signal for a given window size.
    A frame is, usually, 10 milliseconds worth of samples.

    Arguments
    ---------
    pitch : list or array-like
        Pitch values of the signal.
    win_length : int
        Frame length to process.
    ms_per_window : int, optional
        Duration of the window in milliseconds, default is 10ms.

    Returns
    -------
    The creakiness values of the input pitch sequence.

    Example
    -------
    >>> pitch, energy = compute_pitch_energy(signal, sample_rate)
    >>> win_length = 10
    >>> compute_creakiness(pitch, win_length)
    array([0.5, 0.5, 0., 0., 0., ..., 0., 0.5, 1.])
    """

    # Number of frames per window
    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)

    # Calculate frequency ratios between adjacent frames
    ratios = np.array(pitch[1:]) / np.array(pitch[:-1])

    # Check for octave jumps and other frame-to-frame frequency jumps
    octave_up = (ratios > 1.90) & (ratios < 2.10)
    octave_down = (ratios > 0.475) & (ratios < 0.525)
    small_up = (ratios > 1.05) & (ratios < 1.25)
    small_down = (ratios < 0.95) & (ratios > 0.80)

    # Compute creakiness for each frame
    creakiness = octave_up + octave_down + small_up + small_down

    # Calculate the integral image for efficiency
    integral_image = np.concatenate(([0], np.cumsum(creakiness)))
    window_values = integral_image[frames_per_window:].astype(float) - integral_image[:-frames_per_window].astype(float)

    # Pad in front and at the end
    padding_needed = frames_per_window - 1
    front_padding = np.zeros(int(np.floor(padding_needed / 2)))
    tail_padding = np.zeros(int(np.ceil(padding_needed / 2)))
    creak_array = np.concatenate((front_padding, window_values, tail_padding))
    creak_values = creak_array / frames_per_window

    return creak_values


def compute_energy_stability(energy, win_length):
    """Compute energy stability of a signal using a specified window size.

    Arguments
    ---------
    energy : list or array-like
        Energy values of the signal.
    win_length : int
        Frame length to process.

    Returns
    -------
    The energy stability values of the input signal.

    Example
    -------
    >>> pitch, energy = compute_pitch_energy(signal, sample_rate)
    >>> win_length = 10
    >>> compute_energy_stability(energy, win_length)
    array([9., 5.5, 3., ..., 9., 12.5])
    """

    range_count = []
    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)
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
    front_padding = np.zeros(int(np.floor(padding_needed / 2)))
    tail_padding = np.zeros(int(np.ceil(padding_needed / 2)))
    energy_range = np.concatenate((front_padding, window_values, tail_padding))
    energy_stability = energy_range / frames_per_window

    return energy_stability


def compute_pitch_range(pitch, win_length, range_type):
    """Compute pitch range for a given pitch sequence and window size.

    Arguments
    ---------
    pitch : list or array-like
        Pitch values of the signal.
    win_length : int
        Frame length to process.
    range_type : str
        Type of pitch range to compute. Choose between ["f", "n", "w"].
            - "f": Full pitch range
            - "n": Narrow pitch range
            - "w": Wide pitch range

    Returns
    -------
    The pitch range values of the input pitch sequence.

    Example
    -------
    >>> pitch, energy = compute_pitch_energy(signal, sample_rate)
    >>> win_length = 10
    >>> compute_pitch_range(pitch, win_length, range_type="n")
    array([4.5, 7., 7., ..., 7., 7.5])
    """

    range_count = []
    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)
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


    # Compute an integral image for efficiency
    integral_image = np.concatenate(([0], np.cumsum(range_count)))
    window_values = integral_image[frames_per_window:].astype(float) - integral_image[:-frames_per_window].astype(float)

    padding_needed = frames_per_window - 1
    front_padding = np.zeros(int(np.floor(padding_needed / 2)))
    tail_padding = np.zeros(int(np.ceil(padding_needed / 2)))
    pitch_range = np.concatenate((front_padding, window_values, tail_padding))
    pitch_range = pitch_range / frames_per_window

    return pitch_range


def percentilize_pitch(pitch, max_pitch):
    """Compute percentiles for a given pitch sequence and maximum pitch value.
    Instead of using specific Hz values, it maps them to percentiles in the overall distribution.
    This non-linearly scales the input while preserving NaNs.
    It accepts either a row vector or a column vector and returns a column vector.
    The input pitch_points typically range from 50 to around 515. Any points above max_pitch are mapped to NaN.

    Arguments
    ---------
    pitch_points : list or array-like
        Pitch values of the signal.
    max_pitch : int
        Maximum pitch value.

    Returns
    -------
    The pitch percentiles for the input pitch sequence.

    Example
    -------
    >>> pitch, energy = compute_pitch_energy(signal, sample_rate)
    >>> max_pitch = 300
    >>> percentilize_pitch(pitch, max_pitch)
    array([1., NaN, 0.94308943, ..., 0.89430894, 0.88617886])
    """

    rounded = np.asarray(pitch, dtype=np.float64).round()

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


def compute_rate(log_energy, win_length):
    """Compute speaking rate of a signal using a specified window size.
    A frame is, usually, 10 milliseconds worth of samples.

    Arguments
    ---------
    log_energy : list or array-like
        Log energy of the input signal.
    win_length : int
        Frame length to process.

    Returns
    -------
    The speaking rate of the signal.

    Example
    -------
    >>> signal = read_audio('example.wav')
    >>> win_length = 10
    >>> log_energy = compute_log_energy(signal, win_length)
    >>> compute_rate(log_energy, win_length)
    array([-2.1875138, -2.25198406, -2.19078188, ..., -2.15308205, -2.22154751])
    """

    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)      
    
    # Compute inter-frame deltas
    deltas = np.abs(np.diff(log_energy))
    cum_sum_deltas = np.concatenate(([0], np.cumsum(deltas)))
    
    # Adjust the indices to ensure compatible shapes
    window_liveliness = cum_sum_deltas[int(frames_per_window):] - cum_sum_deltas[:-int(frames_per_window)]
    
    # Normalize rate for robustness against recording volume
    silence_mean, speech_mean = find_cluster_means(log_energy)
    scaled_liveliness = (window_liveliness - silence_mean) / (speech_mean - silence_mean)
    
    padding_needed = frames_per_window - 1
    front_padding = np.zeros(int(np.floor(padding_needed / 2)))
    tail_padding = np.zeros(int(np.ceil(padding_needed / 2)))
    rate = np.concatenate((front_padding, scaled_liveliness, tail_padding))

    return rate


def average_of_near_values(values, near_mean, far_mean):
    """Compute the average of values near a specified mean.

    Arguments
    ---------
    values : list or array-like
        The values for which to compute the average.
    near_mean : float
        The mean value for selecting nearby samples.
    far_mean : float
        The mean value for selecting samples that are far away.

    Returns
    -------
    The computed subset average: a float value

    Example
    -------
    >>> values = [1, 2, 3, 4, 5]
    >>> near_mean = 3
    >>> far_mean = 1
    >>> average_of_near_values(values, near_mean, far_mean)
    """

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


def find_cluster_means(values):
    """Find the mean values of two clusters in the input data.

    Arguments
    ---------
    values : list or array-like
        The values for which to find cluster means.

    Returns
    -------
    Tuple containing the low and high cluster means.

    Example
    -------
    >>> values = [1, 2, 3, 4, 5] # example
    >>> find_cluster_means(values)
    (1.0, 5.0)
    """

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

    return previous_low_center, previous_high_center


def compute_epeakness(energy):
    """Compute the peakness of an energy signal.

    Arguments
    ---------
    energy : list or array-like
        The energy values for computing peakness.

    Returns
    -------
    The peakness of the input energy signal.

    Example
    -------
    >>> energy = [0.1, 0.5, 0.8, 0.6, 0.3] usually should be much longer than iFFW * 2.5
    >>> epeakness(energy)
    array([0., 0.22082971, 0.40427613, ..., 0.26171692, 0.])
    """

    iSFW = 6  # in-syllable filter width, in frames
    iFFW = 15  # in-foot filter width, in frames

    energy = np.array(energy)

    # Height normalization
    height = np.sqrt((energy - np.min(energy)) / (np.max(energy) - np.min(energy)))

    # Convolve with Laplacian of Gaussian filters
    inSyllablePeakness = myconv(energy, laplacian_of_gaussian(iSFW), iSFW * 2.5)
    inFootPeakness = myconv(energy, laplacian_of_gaussian(iFFW), iFFW * 2.5)

    # Compute peakness as the product of the above values and height
    epeakness = inSyllablePeakness * inFootPeakness * height

    # Remove negative values
    epeakness[epeakness < 0] = 0

    return epeakness


def compute_ppeakness(pitch):
    """Compute the peakness of a pitch signal.

    Arguments
    ---------
    pitch : list or array-like
        The pitch values for computing peakness.

    Returns
    -------
    The peakness of the input pitch signal.

    Example
    -------
    >>> pitch, energy = compute_pitch_energy(signal, sample_rate)
    >>> ppeakness(pitch)
    array([0., 95.33424693, 0., ..., 337.66122082, 0.])
    """

    ssFW = 10  # stressed-syllable filter width

    # Identify valid pitch values and compute local pitch amount
    pitch = np.array(pitch)
    valid_pitch = pitch > 0
    local_pitch_amount = myconv(1.0 * valid_pitch, triangle_filter(160), 10)

    # Replace NaN values with 0
    pitch[np.isnan(pitch)] = 0

    # Convolve with Laplacian of Gaussian filter
    local_peakness = myconv(pitch, laplacian_of_gaussian(ssFW), 2.5 * ssFW)

    # Compute peakness as the product of the above values and pitch
    ppeakness = local_peakness * local_pitch_amount * pitch

    # Remove negative values
    ppeakness[ppeakness < 0] = 0

    return ppeakness


def compute_spectral_tilt(signal, sample_rate):
    """Compute the spectral tilt of a signal every 10ms.

    Arguments
    ---------
    signal : array-like
        The input signal.
    sample_rate : int
        The sample rate of the signal.

    Returns
    -------
    tilts : array
        The spectral tilt values computed every 10ms.

    Example
    -------
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> compute_spectral_tilt(signal, 16000)
    array([-4.97950109e-21, -1.01362046e-20, ..., -1.08346100e-20, -2.24234884e-21])
    """

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

    win_length = int(0.025 * sample_rate)  # 15ms windows
    overlap = int(0.015 * sample_rate)  # To get a value for every 10 ms, an overlap of 15 is needed
    f, t, spec = spectrogram(signal, fs=sample_rate, window='hann', nperseg=win_length, noverlap=overlap)

    # Build empty matrix for the amplitude values squaring the values
    # obtained from calling spectrogram
    signal_length = len(signal) / sample_rate
    nframes = int(np.floor(signal_length / 0.01))  # number of 10ms windows (frames)
    nfrequencies = len(freq_values)
    amplitudes = np.abs(spec) ** 2  # calculate Power Spectral Density (PSD)
    A = np.zeros((nfrequencies, nframes))
    freq_range = 10  # define the frequency range in Hz

# Fill matrix with the average PSD at each target frequency every 10ms
    for i in range(nfrequencies):
        freq = freq_values[i]
        for j in range(amplitudes.shape[1]):
            freq_idx = (f >= (freq - freq_range)) & (f <= (freq + freq_range))
            
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


def speaking_frames(log_energy):
    """Identify frames with speech based on log energy.

    Arguments
    ---------
    log_energy : array-like
        Log energy values.

    Returns
    -------
    speaking_frames : array
        Boolean array indicating frames with speech.

    Example
    -------
    >>> log_energy = compute_log_energy(signal, win_length)
    >>> speaking_frames(log_energy)
    array([False, False, ...,  True, False])
    """

    # Find silence and speech mean of track using k-means
    silence_mean, speech_mean = find_cluster_means(log_energy)

    # Set the speech/silence threshold closer to the silence mean
    # because the variance of silence is less than that of speech.
    # This is ad hoc; modeling with two Gaussians would probably be better
    threshold = (2 * silence_mean + speech_mean) / 3.0

    sframes = log_energy > threshold

    return sframes


def voiced_unvoiced_ir(log_energy, pitch, win_length):
    """Compute the ratio of average voiced intensity to average unvoiced intensity.

    Arguments
    ---------
    log_energy : array-like
        Log energy values.
    pitch : array-like
        Pitch values.
    win_length : int
        Frame length to process.

    Returns
    -------
    ratio : array
        The ratio of average voiced intensity to average unvoiced intensity.

    Example
    -------
    >>> log_energy = compute_log_energy(signal, win_length)
    >>> pitch, energy = compute_pitch_energy(signal, sample_rate)
    >>> win_length = 20
    >>> voiced_unvoiced_ir(log_energy, pitch, win_length)
    array([0.9, 1.1, ..., 0.8, 0.5])
    """
    
    is_speech = speaking_frames(log_energy)

    voiced_speech_vec = (~np.isnan(pitch) & is_speech)
    unvoiced_speech_vec = (np.isnan(pitch) & is_speech)

    non_voiced_energies_zeroed = voiced_speech_vec * log_energy
    non_unvoiced_energies_zeroed = unvoiced_speech_vec * log_energy

    v_frame_cum_sum = np.concatenate(([0], np.cumsum(non_voiced_energies_zeroed)))
    u_frame_cum_sum = np.concatenate(([0], np.cumsum(non_unvoiced_energies_zeroed)))
    v_frame_cum_count = np.concatenate(([0], np.cumsum(voiced_speech_vec)))
    u_frame_cum_count = np.concatenate(([0], np.cumsum(unvoiced_speech_vec)))

    ms_per_window = 10
    frames_per_window = win_length / ms_per_window
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


def windowize(signal, pitch, win_length):
    """Compute windowed values of input signal.

    Arguments
    ---------
    signal : list or array-like
        The input signal.
    pitch : array-like
        Pitch values.
    win_length : int
        Frame length to process.

    Returns
    -------
    window_values : array
        Summed values over windows of the designated size, centered at 10ms, 20ms, etc.

    Example
    -------
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> win_length = 20
    >>> windowize(signal, pitch, win_length)
    array([2.5, 3.5, ..., 4.5, 5.0])
    """

    integral_image = np.concatenate([[0], np.cumsum(signal)])
    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)
    window_sum = integral_image[frames_per_window:] - integral_image[:-frames_per_window]

    # align so the first value is for the window centered at 10 ms
    padding_needed = frames_per_window - 1
    front_padding = np.zeros(int(np.floor(padding_needed / 2)))
    tail_padding = np.zeros(int(np.ceil(padding_needed / 2)))
    window_values = np.concatenate((front_padding, window_sum, tail_padding))

    return window_values


def window_energy(log_energy, win_length):
    """Compute window energy.

    Arguments
    ---------
    log_energy : array-like
        Log energy values.
    win_length : int
        Frame length to process.

    Returns
    -------
    win_energy : array
        Window energy values.

    Example
    -------
    >>> log_energy = compute_log_energy(signal, win_length)
    >>> win_length = 20
    >>> window_energy(log_energy, win_length)
    array([2.4213e-02, -1.0180e-01, ..., -2.0930e-01, -7.4701e-02])
    """

    integral_image = np.concatenate([[0], np.cumsum(log_energy)])
    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)
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
    padding_needed = frames_per_window - 1
    front_padding = np.zeros(int(np.floor(padding_needed / 2)))
    tail_padding = np.zeros(int(np.ceil(padding_needed / 2)))
    win_energy = np.concatenate((front_padding, scaled_sum, tail_padding))

    return win_energy


def disalignment(epeakness, ppeakness):
    """Compute disalignment estimate (pitch-peak delay).

    Arguments
    ---------
    epeakness : array-like
        An array representing energy peaks.
    ppeakness : array-like
        An array representing pitch peaks.

    Returns
    -------
    array-like
        The disalignment estimate.

    Example
    -------
    >>> epeakness = compute_epeakness(energy)
    >>> ppeakness = compute_ppeakness(pitch)
    >>> disalignment(epeakness, ppeakness)
    array([0.00000000e+00, 0.00000000e+00, ..., 1.63914949e+07, 1.95116422e+07])
    """

    # Find local maxima in epeaky, representing energy peaks
    local_max_epeakness = find_local_max(epeakness, 120)
    
    # Calculate the expected product of local maxima and ppeaky
    expected_product = local_max_epeakness * ppeakness
    
    # Calculate the actual product of epeaky and ppeaky
    actual_product = epeakness * ppeakness
    
    # Compute the disalignment estimate
    disalignment = (expected_product - actual_product) * ppeakness
    
    return disalignment


def find_local_max(values, win_length):
    """Find local maxima in a values input.

    Arguments
    ---------
    values : list or array-like
        A list or array of numeric values.
    win_length : int
        Frame length to process.

    Returns
    -------
    local_maxima : array
        Array with local maxima.

    Example
    -------
    >>> values = [0.2, 0.5, 0.8, 1.2, 1.5]
    >>> win_length = 20
    >>> find_local_max(values, win_length)
    array([-0.00369263, -0.00363159, -0.00350952, ..., -0.00491333])
    """

    # Calculate half the width in frames
    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)
    local_max = np.zeros(len(values))
    
    # Iterate through the values
    for e in range(len(values)):
        start_frame = max(0, e - frames_per_window)
        end_frame = min(e + frames_per_window, len(values))
        
        # Find the maximum value within the defined window
        local_max[e] = max(values[start_frame:end_frame])
    
    return local_max


def myconv(values, kernel, filterHalfWidth):
    """Custom convolution function.

    Arguments
    ---------
    values : list or array-like
        A list or array to be convolved.
    kernel : list or array-like
        The convolution kernel.
    filterHalfWidth : float
        The half-width of the filter.

    Returns
    -------
    convolved_values : array
        The result of the convolution.

    Example
    -------
    >>> values = np.array([0.1, 0.5, 0.8, ..., 0.3, 0.2])
    >>> kernel = np.array([1, -1, 1])
    >>> filterHalfWidth = 1
    >>> myconv(values, kernel, filterHalfWidth)
    array([0.5, 2. , 3. , 3.5, 3.2])
    """

    result = np.convolve(values, kernel, mode='same')
    trimWidth = int(np.floor(filterHalfWidth))

    # Pad with zeros to trimwidth at beginning and end to avoid artifacts
    result[:trimWidth] = 0
    result[-trimWidth:] = 0

    return result


def laplacian_of_gaussian(sigma):
    """Generate Laplacian of Gaussian (LoG) kernel.

    Arguments
    ---------
    sigma : float
        Standard deviation of the Gaussian part of the filter.

    Returns
    -------
    filter : array
        1D Laplacian of Gaussian filter.

    Example
    -------
    >>> sigma = 2.0
    >>> laplacian_of_gaussian(sigma)
    array([-0.   , -0.004, -0.04 , -0.18 , -0.32 , -0.36 , -0.2  ,  0.  ,  0.2  ,  0.36 ,  0.32 ,  0.18 ,  0.04 ,  0.004])
    """

    # Length calculation
    length = int(sigma * 5)

    sigmaSquare = sigma * sigma
    sigmaFourth = sigmaSquare * sigmaSquare

    result = np.zeros(length)
    center = length // 2
    for i in range(length):
        x = i - center
        y = ((x * x) / sigmaFourth - 1 / sigmaSquare) * np.exp((-x * x) / (2 * sigmaSquare))
        result[i] = -y

    return result


def rectangular_filter(win_length):
    """Generate a rectangular filter.

    Arguments
    ---------
    win_length : int
        Frame length of the rectangular filter in milliseconds.

    Returns
    -------
    filter : array
        The rectangular filter.

    Example
    -------
    >>> win_length = 20
    >>> rectangular_filter(win_length)
    array([0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333])
    """

    ms_per_window = 10
    duration_frames = int(np.floor(win_length / ms_per_window))
    filter_values = torch.ones(duration_frames) / duration_frames

    return filter_values


def triangle_filter(win_length):
    """Generate a triangle filter.

    Arguments
    ---------
    win_length : int
        Frame length of the triangle filter in milliseconds.

    Returns
    -------
    filter : array
        The triangle filter.

    Example
    -------
    >>> win_length = 20
    >>> triangle_filter(win_length)
    array([0.2, 0.4, 0.2])
    """

    ms_per_window = 10
    duration_frames = int(np.floor(win_length / ms_per_window))
    duration_frames = int(np.floor(win_length / 10))
    center = int(np.floor(duration_frames / 2))
    filter_values = [center - abs(i - center) for i in range(1, duration_frames + 1)]
    filter_values = np.array(filter_values) / np.sum(filter_values)  # normalize to sum to one

    return filter_values
