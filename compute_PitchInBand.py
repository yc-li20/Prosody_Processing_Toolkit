import numpy as np

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

# Test cases:
# percentiles = Your input percentiles vector
# band_flag = 'h', 'l', 'th', or 'tl' based on the desired band
# window_size_ms = Size of the analysis window in milliseconds
# band_values = compute_pitch_in_band(percentiles, band_flag, window_size_ms)
# print(compute_pitch_in_band([1,1,1,0.9,'NaN',0.8,0.9,1.0,0.95,0.95],'h', 90))
