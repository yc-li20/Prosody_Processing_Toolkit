import numpy as np

def all_tilt_features(perFrameTilts, logEnergy, msPerWindow):
    isSpeakingVec = speakingframes(logEnergy)
    perFrameTilts = trimOrPadIfNeeded(perFrameTilts, isSpeakingVec)
    
    frPerWindow = msPerWindow / 10
    globalMeanOfValids = np.mean([tilt for i, tilt in enumerate(perFrameTilts) if isSpeakingVec[i]])
    
    st = meansOverNonzeros(perFrameTilts, isSpeakingVec, frPerWindow, globalMeanOfValids)
    
    prunedTilts = np.array(perFrameTilts)  # 转换为NumPy数组
    for i, is_speaking in enumerate(isSpeakingVec):
        if not is_speaking:
            prunedTilts[i] = 0
    
    nonzeroTilts = [tilt for tilt in prunedTilts if tilt != 0]
    bincounts, binedges = np.histogram(nonzeroTilts, bins=100)
    
    cumPerc = np.cumsum(bincounts * 1.0 / sum(bincounts))
    veryNegThreshold = ftfp(bincounts, binedges, cumPerc, 0.20)
    nearlyFlatThreshold = ftfp(bincounts, binedges, cumPerc, 0.50)
    ninetyEighthPercentile = ftfp(bincounts, binedges, cumPerc, 0.98)
    
    nearlyFlatFrames = (prunedTilts < 0) & (prunedTilts <= ninetyEighthPercentile) & (prunedTilts > nearlyFlatThreshold)
    middlingNegFrames = (prunedTilts <= nearlyFlatThreshold) & (prunedTilts > veryNegThreshold)
    veryNegFrames = prunedTilts.copy()
    veryNegFrames[veryNegFrames > veryNegThreshold] = 0
    
    tf = meansOverNonzeros(nearlyFlatFrames, isSpeakingVec, frPerWindow, 0)
    tm = meansOverNonzeros(middlingNegFrames, isSpeakingVec, frPerWindow, 0)
    tn = -meansOverNonzeros(veryNegFrames, isSpeakingVec, frPerWindow, 0)
    
    tiltRange = np.zeros(len(perFrameTilts))
    for firstFrame in range(len(perFrameTilts)):
        lastFrame = min(firstFrame + int(frPerWindow) - 1, len(perFrameTilts))
        valid_tilts = [tilt for i, tilt in enumerate(prunedTilts[firstFrame:lastFrame]) if isSpeakingVec[firstFrame + i]]
        if valid_tilts:
            tiltRange[firstFrame] = np.max(valid_tilts) - np.min(valid_tilts)
        else:
            tiltRange[firstFrame] = 0
    
    return st, tiltRange, tf, tm, tn

def trimOrPadIfNeeded(perFrameTilts, isSpeakingVec):
    if len(isSpeakingVec) != len(perFrameTilts):
        lengthDifference = len(perFrameTilts) - len(isSpeakingVec)
        if lengthDifference == -2:
            newPerFrameTilts = [0] + perFrameTilts + [0]
        elif lengthDifference == -1:
            newPerFrameTilts = perFrameTilts + [0]
        elif lengthDifference == 1:
            newPerFrameTilts = perFrameTilts[:-1]
        elif lengthDifference == 2:
            newPerFrameTilts = perFrameTilts[1:-1]
        else:
            raise ValueError('Lengths badly differ.')
        return newPerFrameTilts
    else:
        return perFrameTilts

def speakingframes(log_energy):
    silence_mean, speech_mean = findclustermeans(log_energy)
    threshold = (2 * silence_mean + speech_mean) / 3.0

    log_energy = np.array(log_energy)  # 转换为NumPy数组
    vec = log_energy > threshold
    return vec

def findclustermeans(values):
    max_iterations = 20
    previous_low_center = np.min(values)
    previous_high_center = np.max(values)
    convergence_threshold = (previous_high_center - previous_low_center) / 100

    for _ in range(max_iterations):
        high_center = average_of_near_values(values, previous_high_center, previous_low_center)
        low_center = average_of_near_values(values, previous_low_center, previous_high_center)

        # Check for convergence based on a small threshold
        if (abs(high_center - previous_high_center) < convergence_threshold and
            abs(low_center - previous_low_center) < convergence_threshold):
            return low_center, high_center

        previous_high_center = high_center
        previous_low_center = low_center

    # If max iterations are reached without convergence, issue a warning
    print('findClusterMeans exceeded maxIterations without converging')
    print(previous_high_center)
    print(previous_low_center)
    return previous_low_center, previous_high_center

def average_of_near_values(values, near_mean, far_mean):
    # Returns the averages of all points which are closer to the near mean
    # than to the far mean
    
    # To save time, approximate by taking a sample of 2000 values.
    nsamples = 2000

    if len(values) < 2000:
        samples = values
    else:
        samples = values[::round(len(values) / nsamples)]

    near_mean = float(near_mean)
    far_mean = float(far_mean)

    closer_samples = [sample for sample in samples if abs(sample - near_mean) < abs(sample - far_mean)]

    if len(closer_samples) == 0:
        # If no closer samples are found, return a weighted average
        subset_average = 0.9 * near_mean + 0.1 * far_mean
    else:
        subset_average = np.mean(closer_samples)

    return subset_average

def meansOverNonzeros(data, mask, window_size, global_mean):
    # Calculate the mean of 'data' over non-zero elements within 'mask' using a sliding window of 'window_size'.
    result = np.zeros(len(data))
    for firstFrame in range(len(data)):
        lastFrame = min(firstFrame + int(window_size) - 1, len(data))
        valid_data = [data[i] for i in range(firstFrame, lastFrame) if mask[i] == 1]
        if valid_data:
            result[firstFrame] = np.mean(valid_data)
        else:
            result[firstFrame] = global_mean
    return result

def ftfp(bincounts, binedges, cumPerc, targetPercentile):
    # Find a threshold value in the histogram corresponding to a specific percentile.
    threshold_bin = np.argmin(np.abs(cumPerc - targetPercentile))
    threshold = binedges[threshold_bin]
    return threshold

# Test cases:
# per_frame_tilts = Your tilt data
# log_energy = Logarithm of energy values
# ms_per_window = Duration of the analysis window in milliseconds
# st, tilt_range, tilt_flat, tilt_middling, tilt_neg = all_tilt_features(per_frame_tilts, log_energy, ms_per_window)
# print(all_tilt_features([9,8,7,1,2,3,9,6,6], [1,0,0,1,1,1,0,1,0], 20))
