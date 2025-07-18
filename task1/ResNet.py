import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import torch.optim as optim
import matplotlib.pyplot as plt

def load_dataset(file_dir):
    # 加载训练数据
    train_data = []
    train_labels = []

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    for i in range(1, 6):
        batch = unpickle(os.path.join(file_dir, f'data_batch_{i}'))
        train_data.append(batch[b'data'])       # ndarray: [10000, 3072]
        train_labels += batch[b'labels']

    train_data = np.concatenate(train_data)      # [50000, 3072]
    train_data = train_data.reshape(-1, 3, 32, 32)  # [50000, 3, 32, 32]
    train_data = train_data.astype(np.float32) / 255.0  # u8归一化

    # 加载测试数据
    test_batch = unpickle(os.path.join(file_dir, 'test_batch'))
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_labels = test_batch[b'labels']

    return train_data, train_labels, test_data, test_labels

class cifar_dataset(Dataset):
    def __init__(self, img, label):
        self.images = torch.tensor(img)
        self.labels = torch.tensor(label)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        shortcut = self.shortcut(x)
        out += shortcut   # 残差连接
        out = self.relu(out)
        return out
    
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            ResBlock(64, 64, stride=1),
            ResBlock(64, 64, stride=1)
        )
        self.conv3 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128, stride=1)
        )
        self.conv4 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1)
        )
        self.conv5 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1)
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
    
class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            ResBlock(64, 64, stride=1),
            ResBlock(64, 64, stride=1),
            ResBlock(64, 64, stride=1)
        )
        self.conv3 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128, stride=1),
            ResBlock(128, 128, stride=1),
            ResBlock(128, 128, stride=1)
        )
        self.conv4 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1),
            ResBlock(256, 256, stride=1),
            ResBlock(256, 256, stride=1),
            ResBlock(256, 256, stride=1),
            ResBlock(256, 256, stride=1)
        )
        self.conv5 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1),
            ResBlock(512, 512, stride=1)
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

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

def main():
    dataDir = './cifar-10'
    epochs = 20
    layers = 34
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, train_labels, test_data, test_labels = load_dataset(dataDir)
    train_loader = DataLoader(cifar_dataset(train_data, train_labels), batch_size=64, shuffle=True)
    test_loader = DataLoader(cifar_dataset(test_data, test_labels))

    # 初始化模型
    if layers == 18:
        model = ResNet18(num_classes=10).to(device)
    elif layers == 34:
        model = ResNet34(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    global_acc = 0

    # 初始化记录列表
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # 训练与评估
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 记录数据
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        
        if test_acc > global_acc:
            global_acc = test_acc
            # 保存模型
            torch.save(model.state_dict(), 'base_cifar10.pth')
            print("模型已保存为 base_cifar10.pth")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(test_losses, label='Test Loss', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Test Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(test_accs, label='Test Accuracy', color='red', linewidth=2)
    print(test_accs)
    print(train_losses)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Test Accuracy', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f'RES_{layers}_curves.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()