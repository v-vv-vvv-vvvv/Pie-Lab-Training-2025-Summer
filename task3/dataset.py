import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

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
    train_data = (train_data.astype(np.float32) / 255.0 )  # u8归一化，现在是-1到1

    # 加载测试数据
    test_batch = unpickle(os.path.join(file_dir, 'test_batch'))
    test_data = (test_batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0)
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