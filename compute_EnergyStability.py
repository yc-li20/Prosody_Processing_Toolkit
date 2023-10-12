import numpy as np

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

# Test cases
# energy = Your energy data
# window_size = Your window size in milliseconds, e.g., 100
# energy_range = compute_energy_stability(energy, window_size)
# print(compute_energy_stability([100, 100, 120, 300, 400, 500, 600], 10))
