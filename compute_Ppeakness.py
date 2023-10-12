# Given a vector of pitch-vs-time
# return another vector of the same size.  The value at each point 
# represents the likelihood/strength of a peak there
#
# Pitch peaks have to meet three criteria: 
#  1. the pitch is high, globally
#  2. it's higher than anywhere else in this syllable
#  3. there are enough nearby pitch points for it to be reliable/salient
#
# Pitch may have NaNs.  In general these are tricky, but here we just set
# them all to zero.   Zero times anything is zero, so this means 
# that effectively the convolution will be done skipping over those points


import numpy as np
from scipy.signal import convolve

def ppeakness(pitch_ptile):
    ssFW = 10  # stressed-syllable filter width; could be 12

    # Identify valid pitch values and compute local pitch amount
    pitch_ptile = np.array(pitch_ptile)
    valid_pitch = pitch_ptile > 0
    local_pitch_amount = myconv(1.0 * valid_pitch, triangle_filter(160), 10)

    # Replace NaN values with 0
    pitch_ptile[np.isnan(pitch_ptile)] = 0

    # Convolve with Laplacian of Gaussian filter
    local_peakness = myconv(pitch_ptile, laplacian_of_gaussian(ssFW), 2.5 * ssFW)

    # Compute peakness as the product of the above values and pitch_ptile
    peakness = local_peakness * local_pitch_amount * pitch_ptile

    # Remove negative values (don't care about troughs)
    peakness[peakness < 0] = 0

    return peakness

def myconv(vector, kernel, filterHalfWidth):
    result = convolve(vector, kernel, mode='same')
    trimWidth = int(np.floor(filterHalfWidth))
    result[:trimWidth] = 0
    result[-trimWidth:] = 0
    return result

def laplacian_of_gaussian(sigma):
    # Length calculation
    length = int(sigma * 5)

    sigmaSquare = sigma * sigma
    sigmaFourth = sigmaSquare * sigmaSquare

    vec = np.zeros(length)
    center = length // 2
    for i in range(length):
        x = i - center
        y = ((x * x) / sigmaFourth - 1 / sigmaSquare) * np.exp((-x * x) / (2 * sigmaSquare))
        vec[i] = -y

    return vec

def triangle_filter(window_duration_ms):
    duration_frames = int(np.floor(window_duration_ms / 10))
    center = int(np.floor(duration_frames / 2))
    filter_values = [center - abs(i - center) for i in range(1, duration_frames + 1)]
    filter_values = np.array(filter_values) / np.sum(filter_values)  # normalize to sum to one
    return filter_values


# Validation:
# print(ppeakness([5, 2, 3, 8, 2, 5, ...])) should be much longer than 2.5 * ssFW
