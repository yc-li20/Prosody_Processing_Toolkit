import numpy as np

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


# Test case:
# pitch_points = Your pitch data (e.g., [100, 200, 300, 100, 100, 100, 300, 100, 500, 140, 200, 300, NaN, 500, 600, 300, 500, 700, 100])
# max_pitch = Maximum pitch value (e.g., 500)
# percentiles = percentilize_pitch(pitch_points, max_pitch)
# print(percentilize_pitch([100, 200, 300, 100, 100, 100, 300, 100, 500, 140, 200, 300, 'NaN', 500, 600, 300, 500, 700, 100], 500))