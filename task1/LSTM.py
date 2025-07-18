import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.utils import shuffle
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
# nltk.download('punkt')

# 🔹 步骤 1：加载数据集
def load_imdb_texts(root_dir):
    texts, labels = [], []
    for label_type in ['neg', 'pos']:
        folder = os.path.join(root_dir, label_type)
        for fname in sorted(os.listdir(folder)):
            if fname.endswith('.txt'):
                with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(0 if label_type == 'neg' else 1)
    return texts, labels

# 🔹 步骤 2：构建词表
def build_vocab(texts, max_vocab_size=20000):
    counter = Counter()
    for text in texts:
        tokens = word_tokenize(text.lower())
        counter.update(tokens)
    most_common = counter.most_common(max_vocab_size - 2)  # 预留 <pad> 和 <unk>
    vocab = {'<pad>': 0, '<unk>': 1}
    vocab.update({word: idx + 2 for idx, (word, _) in enumerate(most_common)})
    return vocab

# 数据集类
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.pad_index = vocab['<pad>']
        self.unk_index = vocab['<unk>']
        self.max_len = max_len

        # 统计长度分布
        lengths = [len(text.split()) for text in texts]  # 假设 text 是字符串，如果是已分词则直接 len(text)
        total = len(lengths)
        avg_len = sum(lengths) / total
        max_actual_len = max(lengths)
        sorted_lens = sorted(lengths)
        pct90_len = sorted_lens[int(0.9 * total)]

        print(f"[IMDBDataset] 样本总数: {total}")
        print(f"[IMDBDataset] 平均长度: {avg_len:.2f} 词")
        print(f"[IMDBDataset] 最大长度: {max_actual_len} 词")
        print(f"[IMDBDataset] 90% 样本长度小于: {pct90_len} 词")
        print(f"[IMDBDataset] 当前 max_len 设定为: {max_len}")
        print(len(vocab))

    def __len__(self):
        return len(self.texts)

    def encode(self, text):
        tokens = word_tokenize(text.lower())
        # ids = []  
        # for token in tokens:
        #     if token in self.vocab:
        #         index = self.vocab[token]  
        #     else:
        #         index = self.unk_index     
        #     ids.append(index)            
        # 截断
        # ids = ids[:self.max_len]
        ids = [self.vocab.get(token, self.unk_index) for token in tokens][:self.max_len]
        if len(ids) < self.max_len:
            ids += [self.pad_index] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        text_ids = self.encode(self.texts[idx])
        label = self.labels[idx]
        return torch.tensor(text_ids), torch.tensor(label, dtype=torch.float32)

# 定义模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.dropout(hidden[-1])
        return self.sigmoid(self.fc(out)).squeeze(1)

class MultiLayerLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_index, num_classes=1):
        super(MultiLayerLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        # 三层 LSTM
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True),         # 第一层：输入是 embedding 输出
            nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True, bidirectional=True),        # 第二层
            nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True, bidirectional=True),        # 第三层
        ])
        self.fc = nn.Linear(hidden_dim*2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)  # shape: (batch_size, seq_len, embed_dim)

        out1, _ = self.lstm_layers[0](embedded)
        out2, _ = self.lstm_layers[1](out1)
        out3, _ = self.lstm_layers[2](out2)

        last_hidden = out3[:, -1, :]  # shape: (batch_size, hidden_dim)

        logits = self.fc(last_hidden)  # shape: (batch_size, num_classes)

        return self.sigmoid(logits).squeeze(1)
    
class ResLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_index, num_classes=1):
        super(ResLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        # 三层 LSTM
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(embed_dim, hidden_dim, batch_first=True),   # 第一层：输入是 embedding 输出
            # nn.Linear(embed_dim, hidden_dim),                   # 为了匹配维度
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True),        # 第二层
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True),        # 第三层
        ])
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)  # shape: (batch_size, seq_len, embed_dim)

        out1, _ = self.lstm_layers[0](embedded)
        # x1 = self.lstm_layers[1](embedded)
        # out1 = out1 + x1

        out2, _ = self.lstm_layers[1](out1)
        out2 = out2 + out1

        out3, _ = self.lstm_layers[2](out2)
        out3 = out3 + out2

        last_hidden = out3[:, -1, :]  # shape: (batch_size, hidden_dim)

        logits = self.fc(last_hidden)  # shape: (batch_size, num_classes)

        return self.sigmoid(logits).squeeze(1)

def evaluate(model, loader, device, criterion):
    model.eval
    correct, total = 0, 0
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            predicted = (preds >= 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            loss = criterion(preds, y_batch)
            total_loss += loss.item()
    return total_loss/len(loader), correct/total

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
    # 超参数
    MAX_LENGTH = 500
    BATCH_SIZE = 64
    EMBED_DIM = 128
    HIDDEN_DIM = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    train_texts, train_labels = load_imdb_texts('./aclImdb/train')
    test_texts, test_labels = load_imdb_texts('./aclImdb/test')
    vocab = build_vocab(train_texts)
    pad_idx = vocab['pad']

    train_texts, train_labels = shuffle(train_texts, train_labels)
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, MAX_LENGTH)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MultiLayerLSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, pad_idx).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # 🔹 步骤 6：训练
    print("Start training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for x_batch, y_batch in loop:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(x_batch)
            
            pred_labels = (preds >= 0.5).float()
            correct += (pred_labels == y_batch).sum().item()
            total += y_batch.size(0)
            loss = criterion(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        print(correct, total)
        test_loss, test_acc = evaluate(model, test_loader, DEVICE, criterion)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_loader):.4f}, Test Acc: {test_acc}")

    draw(train_losses, train_accs, test_losses, test_accs, 'BLSTM')


if __name__ == '__main__':
    main()