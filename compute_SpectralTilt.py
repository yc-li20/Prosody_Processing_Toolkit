import numpy as np
from scipy.signal import spectrogram
from scipy.stats import linregress

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
