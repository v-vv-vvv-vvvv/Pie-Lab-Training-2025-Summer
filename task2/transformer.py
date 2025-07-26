import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

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

class imgEmbedding(nn.Module):
    def __init__(self, img_rows, patches, embed_dim, patch_rows):
        super().__init__()
        self.patches = patches
        self.patch = nn.Conv2d(3, embed_dim, patch_rows, patch_rows)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, p):
        p = self.patch(p)
        p = p.flatten(2)
        p = p.transpose(1, 2)
        return self.norm(p)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"

        # 显式定义每个head的Q/K/V权重
        self.W_q = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, self.head_dim)) for _ in range(num_heads)
        ])
        self.W_k = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, self.head_dim)) for _ in range(num_heads)
        ])
        self.W_v = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, self.head_dim)) for _ in range(num_heads)
        ])

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        heads_outputs = []
        for i in range(self.num_heads):
            Q = x @ self.W_q[i] 
            K = x @ self.W_k[i]
            V = x @ self.W_v[i]

            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, seq_len, seq_len]
            attn = torch.softmax(scores, dim=-1)
            out = attn @ V  # [batch, seq_len, head_dim]
            heads_outputs.append(out)

        # 拼接所有head的输出
        multihead_out = torch.cat(heads_outputs, dim=-1)  # [batch, seq_len, embed_dim]
        return self.out_proj(multihead_out)

class VITBlock(nn.Module):
    def __init__(self, embed_dim, heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VIT(nn.Module):
    def __init__(self, img_rows=32, patch_rows=8, num_classes=10,
                 embed_dim=192, depth=6, heads=3, mlp_dim=768):
        super().__init__()
        num_patches = (img_rows // patch_rows) ** 2
        # 语义编码
        self.patch_embed = imgEmbedding(img_rows, num_patches, embed_dim, patch_rows)
        
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        # Transformer Blocks, 一共12层
        self.blocks = nn.Sequential(*[
            VITBlock(embed_dim, heads, mlp_dim) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, embed_dim)

        cls_tokens = self.class_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, embed_dim)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])  # 只用cls token做分类

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

def draw(train_losses,train_accs,test_losses,test_accs,filename):
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
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')

def main():
    dataDir = './cifar/cifar-10'
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, train_labels, test_data, test_labels = load_dataset(dataDir)
    train_loader = DataLoader(cifar_dataset(train_data, train_labels), batch_size=64, shuffle=True)
    test_loader = DataLoader(cifar_dataset(test_data, test_labels))

    # print(len(train_data))

    model = VIT(img_rows=32, patch_rows=8, num_classes=10).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    checkpoint_path = 'checkpoint-16.pth'
    start_epoch = 0  # 默认从头开始

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = checkpoint.get('best_acc', 0.0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"🔁 Loaded checkpoint from epoch {start_epoch}, best accuracy: {best_acc:.4f}")
    else:
        print("🆕 No checkpoint found, starting from scratch.")

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    best_acc = 0.0  # 记录最优模型准确率
    best_model_path = "best_model.pth"

    # 训练与评估循环
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), acc=correct / total)

        # 每轮结束后评估
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }, 'checkpoint-16.pth')
            print(f"✅ Best model saved with accuracy: {best_acc:.4f}")

    draw(train_losses, train_accs, test_losses, test_accs, 'VIT-16-1')

if __name__=="__main__":
    main()