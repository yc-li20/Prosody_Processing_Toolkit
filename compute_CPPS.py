# calculate Smoothed Cepstral Peak Prominence (CPPS)

import numpy as np
from scipy.signal import spectrogram, lfilter

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

# Test
# sample_rate = 100  # Sample rate in Hz
# signal = np.random.rand(100)  # Example signal
# print(compute_CPPS(signal, sample_rate))