import numpy as np

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

# Test cases
# pitch = Your pitch data, e.g., [1, 1, 1, 1.1, 0.8, 1.0, 0.9, 1, 1, 1, 1.01, 1.02, 1]
# window_size_ms = Your window size in milliseconds
# creak_values = compute_creakiness(pitch, window_size_ms)
# print(compute_creakiness(pitch, 10))
