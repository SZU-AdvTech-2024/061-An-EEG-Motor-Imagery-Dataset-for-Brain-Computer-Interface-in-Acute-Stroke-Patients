import numpy as np
import scipy.signal as signal
import torch.nn as nn


class ComputeCovariance:
    def __init__(self, fs: int = 1000):
        self.fs = fs
        
        # 定义频带
        self.filter_banks = [
            (8, 12), (9, 13), (10, 14), (11, 15), (12, 16), (13, 17),
            (14, 18), (15, 19), (16, 20), (17, 21), (18, 22), (19, 23),
            (20, 24), (21, 25), (22, 26), (23, 27), (24, 28), (25, 29), (26, 30)
        ]
        
        # 定义时间窗口（秒）
        self.time_windows = [
            (0, 1), (0.5, 1.5), (1, 2)
        ]
    
    def apply_bandpass_filter(self, data: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """
        对信号进行带通滤波
        :param data: 输入EEG数据，形状为 (num_channels, num_samples)
        :param low_freq: 低频
        :param high_freq: 高频
        :return: 滤波后的信号
        """
        nyquist = 0.5 * self.fs  # 奈奎斯特频率
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        return filtered_data

    def calculate_covariance(self, data: np.ndarray) -> np.ndarray:
        """
        计算协方差矩阵
        :param data: 输入数据，形状为 (num_channels, num_time_steps)
        :return: 协方差矩阵，形状为 (num_channels, num_channels)
        """
        # 对时间维度去均值
        mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - mean  # 去均值
        cov_matrix = np.dot(data_centered, data_centered.T) / (data.shape[1] - 1)
        return cov_matrix

    def process_eeg(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        对EEG数据进行预处理
        :param eeg_data: 输入原始EEG数据，形状为 (num_channels, num_samples)
        :return: 计算得到的协方差矩阵集，形状为 (num_windows, num_bands, num_channels, num_channels)
        """
        cov_matrices = []

        for low_freq, high_freq in self.filter_banks:
            # 频带滤波
            filtered_data = self.apply_bandpass_filter(eeg_data, low_freq, high_freq)
            # 计算协方差矩阵
            cov_matrix = self.calculate_covariance(filtered_data)
            cov_matrices.append(cov_matrix)
        
        return np.array(cov_matrices)
    
    def __call__(self, x):
        data = self.process_eeg(x)
        return data