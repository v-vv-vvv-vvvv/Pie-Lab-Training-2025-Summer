import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_test_data(file_dir):
    # 加载训练数据
    # train_data = []
    # train_labels = []

    # for i in range(1, 6):
    #     batch = unpickle(os.path.join(file_dir, f'data_batch_{i}'))
    #     train_data.append(batch[b'data'])       # ndarray: [10000, 3072]
    #     train_labels += batch[b'labels']

    # train_data = np.concatenate(train_data)      # [50000, 3072]
    # train_data = train_data.reshape(-1, 3, 32, 32)  # [50000, 3, 32, 32]
    # train_data = train_data.astype(np.float32) / 255.0  # u8归一化

    # 加载测试数据
    test_batch = unpickle(os.path.join(file_dir, 'test_batch'))
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_labels = test_batch[b'labels']

    return test_data, test_labels

def load_label_names(data_dir):
    meta = unpickle(os.path.join(data_dir, 'batches.meta'))
    label_names = [label.decode('utf-8') for label in meta[b'label_names']]
    return label_names

class cnn_model(nn.Module):
    def __init__(self, num_classes=10):
        super(cnn_model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(PlainBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
    
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            PlainBlock(64, 64, stride=1),
            PlainBlock(64, 64, stride=1)
        )
        self.conv3 = nn.Sequential(
            PlainBlock(64, 128, stride=2),
            PlainBlock(128, 128, stride=1)
        )
        self.conv4 = nn.Sequential(
            PlainBlock(128, 256, stride=2),
            PlainBlock(256, 256, stride=1)
        )
        self.conv5 = nn.Sequential(
            PlainBlock(256, 512, stride=2),
            PlainBlock(512, 512, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def predict(model = cnn_model(num_classes = 10), path = './cnn_cifar10.pth', img = None):
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()

    inputs = torch.tensor(img).float()  # [5, 3, 32, 32]
    outputs = model(inputs)
    _, pred_labels = torch.max(outputs, dim=1)
    return pred_labels

def main():
    data_dir = './cifar-10'
    model_path = './cnn_cifar10.pth'

    # 加载数据和标签名
    data, labels = load_test_data(data_dir)
    label_names = load_label_names(data_dir)

    # 随机选择5张图像
    indices = random.sample(range(len(data)), 5)
    images = data[indices]
    gt_labels = [labels[i] for i in indices]

    pred_labels = predict(img = images)

    def visualize_predictions(images, preds, labels, label_names, title):
        plt.figure(figsize=(12, 4))
        for i in range(len(images)):
            img = images[i].transpose(1, 2, 0)  # CHW -> HWC
            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.title(f"Pred: {label_names[preds[i]]}\nTrue: {label_names[labels[i]]}")
            plt.axis('off')
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        

    # 显示图像和预测结果
    visualize_predictions(images, pred_labels.tolist(), gt_labels, label_names, "CNN result")

    resnet_labels = predict(ResNet18(num_classes=10), './ResNetBase_cifar10.pth', images)
    visualize_predictions(images, resnet_labels.tolist(), gt_labels, label_names, "ResNetBase result")

if __name__ == '__main__':
    main()