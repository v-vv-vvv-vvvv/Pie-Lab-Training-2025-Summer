import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.utils import shuffle
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
# nltk.download('punkt')
import matplotlib.pyplot as plt

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

# 预制模型
# class LSTMClassifier(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, pad_index):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(hidden_dim, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         embedded = self.embedding(x)
#         _, (hidden, _) = self.lstm(embedded)
#         out = self.dropout(hidden[-1])
#         return self.sigmoid(self.fc(out)).squeeze(1)

class CustomLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomLSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # LSTM 门控
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size)  # forget gate
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size)  # input gate
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size)  # output gate
        self.Wc = nn.Linear(input_size + hidden_size, hidden_size)  # candidate cell state

        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)

        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input_seq):
        """
        input_seq: Tensor of shape [batch_size, seq_len, input_size]
        return: logits of shape [batch_size, output_size]
        """
        batch_size, seq_len, _ = input_seq.size()
        device = input_seq.device

        # 初始化 h_0, c_0
        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=device)

        for t in range(seq_len):
            x_t = input_seq[:, t, :]  # 取第 t 个时间步的输入 [batch_size, input_size]
            combined = torch.cat([x_t, h_t], dim=1)  # 拼接输入和上一个 h_t → [batch_size, input+hidden]

            f_t = self.sigmoid(self.Wf(combined))         # forget gate
            i_t = self.sigmoid(self.Wi(combined))         # input gate
            o_t = self.sigmoid(self.Wo(combined))         # output gate
            c_hat_t = self.tanh(self.Wc(combined))        # candidate cell state

            c_t = f_t * c_t + i_t * c_hat_t                # 更新 cell state
            h_t = o_t * self.tanh(c_t)                     # 更新 hidden state

        # 取最后一个时间步的 hidden state，送入分类层
        logits = self.output_layer(h_t)  # shape: [batch_size, output_size]
        return logits

def evaluate(model, loader, device, criterion, embedding):
    model.eval
    correct, total = 0, 0
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # 对输入做 embedding
            embedded_input = embedding(x_batch)  # shape: [batch_size, seq_len, input_size]

            # 前向传播
            preds = model(embedded_input).squeeze(1)  # shape: [batch_size]

            # preds = model(x_batch)
            predicted = (torch.sigmoid(preds) >= 0.5).float()
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

    # 初始化 embedding 和模型（只做一次）
    embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=EMBED_DIM)
    model = CustomLSTMClassifier(input_size=EMBED_DIM, hidden_size=HIDDEN_DIM, output_size=1)
    model.to(DEVICE)
    embedding.to(DEVICE)

    # 优化器要包含 embedding 的参数
    optimizer = torch.optim.Adam(list(model.parameters()) + list(embedding.parameters()), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []


    # 🔹 开始训练
    print("Start training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for x_batch, y_batch in loop:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            # 对输入做 embedding
            embedded_input = embedding(x_batch)  # shape: [batch_size, seq_len, input_size]

            # 前向传播
            preds = model(embedded_input).squeeze(1)  # shape: [batch_size]
            pred_labels = (torch.sigmoid(preds) >= 0.5).float()
            correct += (pred_labels == y_batch).sum().item()
            total += y_batch.size(0)

            # 计算损失
            loss = criterion(preds, y_batch)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        print(correct, total)
        test_loss, test_acc = evaluate(model, test_loader, DEVICE, criterion, embedding)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_loader):.4f}, Test Acc: {test_acc}")

    draw(train_losses, train_accs, test_losses, test_accs, 'LSTM')
    # # 🔹 步骤 7：测试
    # model.eval()
    # correct, total = 0, 0
    # with torch.no_grad():
    #     for x_batch, y_batch in test_loader:
    #         x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
    #         preds = model(x_batch)
    #         predicted = (preds >= 0.5).float()
    #         correct += (predicted == y_batch).sum().item()
    #         total += y_batch.size(0)

    # print(f"Test Accuracy: {correct / total:.4f}")

if __name__ == '__main__':
    main()