# Inspired by the need to estimate pitch-peak delay, this script focuses on
# measuring disalignment without worrying about whether it's shifted forward
# or backward. It's challenging to determine the exact shift without a
# stressed-syllable oracle. Conceptually what's happening is that, at each energy time point, we
# gather evidence for that being aligned with pitch timepoints offset
# to the left and to the right.
#
# disalignments are only significant if they occur at a peak.

import numpy as np

def disalignment(epeaky, ppeaky):
    # Find local maxima in epeaky, representing energy peaks
    local_max_epeak = find_local_max(epeaky, 120)
    
    # Calculate the expected product of local maxima and ppeaky
    expected_product = local_max_epeak * ppeaky
    
    # Calculate the actual product of epeaky and ppeaky
    actual_product = epeaky * ppeaky
    
    # Compute the disalignment estimate
    estimate = (expected_product - actual_product) * ppeaky
    
    return estimate

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


# Example usage:
# To test the find_local_max function, use:
# find_local_max([1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 2, 4, 5, 7, 1, 1, 1, 1, 2], 60)
# It should return the local maxima in the given list.
