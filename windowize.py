import numpy as np

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

# Test cases:
# print(windowize(np.array([0, 1, 1, 1, 2, 3, 3, 3, 1, 1, 1, 2]), 20))
# windowize(np.array([0, 1, 1, 1, 2, 3, 3, 3, 1, 1, 1, 2]), 30)
