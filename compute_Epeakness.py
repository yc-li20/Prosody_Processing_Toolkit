# Input: a vector of energy-vs-time
#    positive and typically ranges from about 5 to 10 
# Output: a vector of the same size whose
#    value at each point represents the likelihood/strength of a peak there
#
# To count as a peak, three properties are important:
# 1. the point is globally high-energy, relative to the whole dialog
# 2. the point is high footwise, that is, it's high relative to
#    the average over the local few syllables, to see if this
#    one is a stressed syllable in the midst of unstressed ones 
# 3. the point is high in its syllable one.

import numpy as np
from scipy.signal import convolve

def epeakness(vec):
    iSFW = 6  # in-syllable filter width, in frames
    iFFW = 15  # in-foot filter width, in frames

    vec = np.array(vec)

    # Height normalization
    height = np.sqrt((vec - np.min(vec)) / (np.max(vec) - np.min(vec)))

    # Convolve with Laplacian of Gaussian filters
    inSyllablePeakness = myconv(vec, laplacian_of_gaussian(iSFW), iSFW * 2.5)
    inFootPeakness = myconv(vec, laplacian_of_gaussian(iFFW), iFFW * 2.5)

    # print(inSyllablePeakness, inFootPeakness)

    # Compute peakness as the product of the above values and height
    peakness = inSyllablePeakness * inFootPeakness * height

    # Remove negative values (we don't consider troughs when aligning)
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


# Validation:
# Eyeballing some graphs, it seems that for energy, this is a 
# fairly decent syllable detector, at least for the stressed
# syllables, which are decently long and decently separated
# from their neighbors. We don't consider the others, the slurred ones.
# print(epeakness([5, 2, 3, 8, 2, 5, ...])) usually should be much longer than iFFW * 2.5