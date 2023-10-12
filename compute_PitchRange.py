import numpy as np

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

# Test cases
# pitch = Your pitch data
# window_size = Your window size in milliseconds, e.g., 100
# range_type = 'f' for flat, 'n' for narrow, 'w' for wide
# pitch_range = compute_pitch_range(pitch, window_size, range_type)
# print(compute_pitch_range([100, 200, 300, 100, 100], 10, 'f'))
