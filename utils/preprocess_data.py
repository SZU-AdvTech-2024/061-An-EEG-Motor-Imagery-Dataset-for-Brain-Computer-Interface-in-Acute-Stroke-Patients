import numpy as np
from utils.notch_filter import notch_filter
from utils.bandpass_filter import bandpass_filter

def sliding_window_split(data, labels, sampling_rate, window_size, step_size):
    eeg_data = []
    eeg_labels = []
    window_sample_point = int(sampling_rate * window_size)
    step_sample_point = int(sampling_rate * step_size)

    for i in range(data.shape[0]):
        for j in range(0, data.shape[2], step_sample_point):
            if j + window_sample_point <= data.shape[2]:
                eeg_data.append(data[i, :, j : j + window_sample_point])
                eeg_labels.append(labels[i])

    return eeg_data, eeg_labels

def preprocess_data(mat_data, split_with_window=False, sampling_rate=250, low_freq=4, high_freq=50, window_size=1, step_size=1):
    """
    对脑电数据进行预处理

    returns:
    
    """
    eeg_data = mat_data['eeg']['rawdata'][0][0]

    labels = mat_data['eeg']['label'][0][0]
    labels = labels[:,0]

    eeg_data = np.transpose(eeg_data, (0, 2, 1))

    data = eeg_data.reshape(-1, eeg_data.shape[2])

    # 保留有用电极
    channel = [i for i in range(30) if i != 17] # 第18是参考电极， 30，31是eog, 32是事件电极
    # print(channel)

    # 获取事件电极进行触发时的数据点
    trigger = np.where(data[:, 32] == 2)[0]
    # print(trigger)

    eeg = np.zeros((40, 29, 2000))

    for i in range(len(trigger)):
        R = data[int(trigger[i]-800):int(trigger[i]-800+2800), channel].T
        # print(R.shape)
        notch_data = notch_filter(R, sampling_rate, 50)

        bandpass_data = bandpass_filter(notch_data, sampling_rate, low_freq, high_freq)
        eeg[i,:,:] = bandpass_data[:, 800:2800]
    # print(eeg)

    if split_with_window:
        eeg, labels = sliding_window_split(eeg, labels, sampling_rate, window_size, step_size)

    return eeg, labels
    