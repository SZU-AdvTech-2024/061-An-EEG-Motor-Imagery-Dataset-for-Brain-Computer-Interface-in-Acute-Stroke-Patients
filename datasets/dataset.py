from torch.utils.data import Dataset
import torch

class StrokePatientDataset(Dataset):
    def __init__(self, eeg_data, labels, data_transform=None, label_transform=None):
        self.eeg_data = eeg_data
        self.labels = labels
        self.data_transform = data_transform
        self.label_transform = label_transform

        if self.data_transform:
            self.eeg_data = [self.data_transform(eeg=data)['eeg'] for data in self.eeg_data]

        if self.label_transform:
            self.labels = [self.label_transform(y=label)['y'] for label in self.labels]

        self.labels = [label - 1 for label in self.labels]

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, index):
        return self.eeg_data[index], self.labels[index]

class StrokePatientDataset1(Dataset):
    def __init__(self, eeg_data, labels, data_transform=None, label_transform=None):
        self.eeg_data = eeg_data
        self.labels = labels
        self.data_transform = data_transform
        self.label_transform = label_transform
        # print(self.eeg_data)
        if self.data_transform:
            eeg_data = [self.data_transform(data) for data in self.eeg_data]
        self.eeg_data = torch.stack(eeg_data)
        if self.label_transform:
            self.labels = [self.label_transform(y=label)['y'] for label in self.labels]

        self.labels = [label - 1 for label in self.labels]

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, index):
        return self.eeg_data[index], self.labels[index]
