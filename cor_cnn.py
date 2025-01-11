import os
import torch
import pandas as pd
import numpy as np
from scipy.io import loadmat
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torcheeg import transforms
from datasets.dataset import StrokePatientDataset1
from utils.preprocess_data import preprocess_data
from utils.early_stopping import EarlyStopping
from sklearn.model_selection import GroupKFold
from transforms.compute_covariance import ComputeCovariance
import torchvision.transforms as transforms
from typing import Tuple
import torch.nn.functional as F


class CCNN(nn.Module):
    def __init__(self, in_channels: int = 4, grid_size: Tuple[int, int] = (9, 9), num_classes: int = 2, dropout: float = 0.5):
        super(CCNN, self).__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(self.in_channels, 32, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(32, 64, kernel_size=4, stride=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(64, 128, kernel_size=4, stride=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(128, 32, kernel_size=4, stride=1), nn.ReLU())

        self.lin1 = nn.Sequential(
            nn.Linear(self.grid_size[0] * self.grid_size[1] * 32, 1024),
            nn.SELU(),
            nn.Dropout2d(self.dropout)
        )
        self.lin2 = nn.Linear(1024, self.num_classes)

    def feature_dim(self):
        return self.grid_size[0] * self.grid_size[1] * 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)

        x = self.lin1(x)
        x = self.lin2(x)
        return x
    
writer = SummaryWriter(log_dir='/root/fm/stroke/runs/dgcnn')
seed = 42
torch.manual_seed(seed)

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    class Data:
        RAW_DATA_PATH = "/root/autodl-tmp/fm/datasets/stroke_data/sourcedata"
        SAMPLING_RATE = 500
        LOW_FREQ = 4
        HIGH_FREQ = 50
        WINDOW_SIZE = 1
        STEP_SIZE = 1

    class Training:
        TRAIN_BATCH_SIZE = 16
        TEST_BATCH_SIZE = 8
        NUM_EPOCHS = 150
        LEARNING_RATE = 0.1

def load_preprocess_data(folder_name):
    file_path = os.path.join(Config.Data.RAW_DATA_PATH, folder_name, folder_name + "_task-motor-imagery_eeg.mat")
    mat_data = loadmat(file_path)
    
    eeg_data, labels = preprocess_data(mat_data, split_with_window=True, sampling_rate=Config.Data.SAMPLING_RATE, 
                                       low_freq=Config.Data.LOW_FREQ, high_freq=Config.Data.HIGH_FREQ, 
                                       window_size=Config.Data.WINDOW_SIZE, step_size=Config.Data.STEP_SIZE)

    data_transform = transforms.Compose([
        ComputeCovariance(fs=500),
        transforms.Lambda(lambda x: torch.tensor(x))
    ])

    dataset = StrokePatientDataset1(eeg_data, labels, data_transform=data_transform)
    print(len(dataset))
    return dataset


def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, scheduler):
    best_val_accuracy = 0.0
    # early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.001)


    for epoch in range(Config.Training.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(Config.DEVICE).float(), targets.to(Config.DEVICE)
            # print(111, inputs.shape)
            optimizer.zero_grad()  # 清除以前的梯度
            outputs = model(inputs)  # 前向传播

            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{Config.Training.NUM_EPOCHS}], Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
 
        scheduler.step(avg_train_loss)

        # 记录训练损失和准确率
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.flush()

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(Config.DEVICE).float(), targets.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
        #     torch.save(model.state_dict(), 'best_model.pth')
    print(f'Best Validation Accuracy: {best_val_accuracy:.2f}%')
    return best_val_accuracy


def cross_validation(dataset, k=4):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    accuracies = [] # 存储每一折的准确率

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}/{k}")
        
        # 使用 Subset 从 Dataset 中提取训练集和验证集
        train_subset = Subset(dataset, train_idx)  # 训练集
        val_subset = Subset(dataset, val_idx)  # 验证集

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        model = CCNNWithLogMap(num_classes=2, in_channels=19, grid_size=(29, 29), dropout=0.3).to(Config.DEVICE)
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.Training.LEARNING_RATE)

        # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        best_val_accuracy = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, scheduler)

        # 存储每一折的准确率
        accuracies.append(best_val_accuracy)
        
    return np.mean(accuracies), np.std(accuracies)

def main():
    raw_data_dir_path = "/root/autodl-tmp/fm/datasets/stroke_data/sourcedata"
    raw_data_dir = sorted(os.listdir(raw_data_dir_path))

    results = []

    for folder_name in raw_data_dir:
        dataset = load_preprocess_data(folder_name)

        mean_accuracy, std_accuracy = cross_validation(dataset)

        # 将每组结果存入字典
        results.append({
            "Subject": f"Subject_{folder_name}",
            "Mean_Accuracy": mean_accuracy,
            "Std_Accuracy": std_accuracy
        })

    df = pd.DataFrame(results)

    # 保存为 CSV 文件
    output_file = "/root/fm/stroke/results_ccnn.csv"
    df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()