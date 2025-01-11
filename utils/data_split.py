import numpy as np


def data_split(eeg_data, labels, time_split):
    num_splits = eeg_data.shape[0] // time_split  # 每个试次会被切分成多少段
    # 初始化存储切分后的数据
    split_data = []
    split_labels = []

    # 对每个试次进行切分
    for i in range(eeg_data.shape[3]):  # 遍历所有试次
        tmp = []
        for j in range(num_splits):  # 每个试次按时间切分
            start = j * time_split
            end = (j + 1) * time_split
            
            # 切分当前试次的 EEG 数据和对应标签
            segment = eeg_data[start:end, :, :, i]  # 获取当前时间段的 EEG 数据，形状为 (500, 29)
 
            split_data.append(segment)  # 存储切分后的数据
            split_labels.append(labels[i])  # 存储对应的标签

    return split_data, split_labels


def data_split_overlap(eeg_data, labels, time_split, overlap):
    """
    切分数据并增加重叠部分
    :param eeg_data: 原始 EEG 数据，形状为 (num_samples, num_channels, num_trials)
    :param labels: 标签，形状为 (num_trials,)
    :param time_split: 每个切分段的时长，单位是样本点数（例如，1秒即500个采样点）
    :param overlap: 重叠的时长，单位是样本点数（例如，0.5秒即250个采样点）
    :return: 切分后的数据和标签
    """
    num_splits = (eeg_data.shape[0] - time_split) // (time_split - overlap) + 1  # 计算有多少个切分段
    
    # 初始化存储切分后的数据
    split_data = []
    split_labels = []

    # 对每个试次进行切分
    for i in range(eeg_data.shape[2]):  # 遍历所有试次
        for j in range(num_splits):  # 每个试次按时间切分
            start = j * (time_split - overlap)  # 每个切分段的开始位置，加入重叠部分
            end = start + time_split  # 计算切分段的结束位置
            
            # 切分当前试次的 EEG 数据和对应标签
            segment = eeg_data[start:end, :, i]  # 获取当前时间段的 EEG 数据，形状为 (500, 29)
            split_data.append(segment)  # 存储切分后的数据
            split_labels.append(labels[i])  # 存储对应的标签
    
    return split_data, split_labels
