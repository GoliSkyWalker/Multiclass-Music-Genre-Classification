
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import wandb
import glob
import random

class SequentialFeatureDataset(Dataset):
    def __init__(self, mfcc_dir, wavnet_dir, label_map):
        self.samples = []
        self.labels = []
        max_len_mfcc = 0
        max_len_wavnet = 0
        for genre in os.listdir(mfcc_dir):
            genre_mfcc_path = os.path.join(mfcc_dir, genre)
            genre_wavnet_path = os.path.join(wavnet_dir, genre)
            if not os.path.isdir(genre_mfcc_path):
                continue
            for mfcc_file in glob.glob(os.path.join(genre_mfcc_path, '*.npy')):
                base = os.path.basename(mfcc_file)
                wavnet_file = os.path.join(genre_wavnet_path, base)
                if not os.path.exists(wavnet_file):
                    continue
                mfcc = np.load(mfcc_file)  # shape: (13, num_windows)
                wavnet = np.load(wavnet_file).reshape(16,-1) # shape: (16, num_windows)
                self.samples.append((mfcc, wavnet))
                self.labels.append(label_map[genre])
                max_len_mfcc = max(max_len_mfcc, mfcc.shape[1])
                max_len_wavnet = max(max_len_wavnet, wavnet.shape[1])

                self.samples.append((mfcc, wavnet))
                self.labels.append(label_map[genre])
        self.max_len_mfcc = max_len_mfcc  
        self.max_len_wavnet = max_len_wavnet
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        mfcc, wavnet = self.samples[idx]

        # Calculate padding needed
        mfcc_pad_len = self.max_len_mfcc - mfcc.shape[1]
        wavnet_pad_len = self.max_len_wavnet - wavnet.shape[1]

        # Pad the MFCC features
        mfcc_padded = np.pad(mfcc, ((0, 0), (0, mfcc_pad_len)), 'constant')

        # Pad the wavnet features (assuming they need to be the same length)
        wavnet_padded = np.pad(wavnet,((0, 0), (0, wavnet_pad_len)), 'constant')


        label = self.labels[idx]
        return torch.tensor(mfcc_padded, dtype=torch.float32), torch.tensor(wavnet_padded, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class RNNFeatureModel(nn.Module):
    def __init__(self, mfcc_dim=13, wavnet_dim=16, rnn_hidden=64, nn_hidden=100, num_classes=10):
        super().__init__()
        self.mfcc_rnn = nn.GRU(mfcc_dim, rnn_hidden, batch_first=True)
        self.wavnet_rnn = nn.GRU(wavnet_dim, rnn_hidden, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden * 2, nn_hidden)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(nn_hidden, num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, mfcc_seq, wavnet_seq):
        mfcc_seq = mfcc_seq.transpose(1, 2) # from (batch, features, seq_len) to (batch, seq_len, features)
        wavnet_seq = wavnet_seq.transpose(1, 2)
        _, mfcc_out = self.mfcc_rnn(mfcc_seq) # shape: (1, batch, rnn_hidden)
        _, wavnet_out = self.wavnet_rnn(wavnet_seq)
        mfcc_vec = mfcc_out.squeeze(0)
        wavnet_vec = wavnet_out.squeeze(0)
        concat = torch.cat([mfcc_vec, wavnet_vec], dim=1)
        x = self.fc1(concat)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class TorchRNNExperiment:
    def __init__(self, args):
        self.args = args
        seed = getattr(args, 'seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.epochs = getattr(args, 'rnn_epochs', 101)
        self.batch_size = getattr(args, 'rnn_batch_size', 8)
        self.lr = getattr(args, 'rnn_lr', 1e-5)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_map = self._get_label_map()
        self.model = RNNFeatureModel(num_classes=len(self.label_map)).to(self.device)
        self._load_data()
    def _get_label_map(self):
        genres = sorted(os.listdir('./data/features/sequential/mfcc'))
        return {genre: i for i, genre in enumerate(genres)}
    def _load_data(self):
        dataset = SequentialFeatureDataset('./data/features/sequential/mfcc', './data/features/sequential/wavenet', self.label_map)
        indices = np.arange(len(dataset))
        train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, shuffle=True)
        self.train_dataset = torch.utils.data.Subset(dataset, train_idx)
        self.test_dataset = torch.utils.data.Subset(dataset, test_idx)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
    def train(self):
        wandb.init(project=self.args.wandb_project, config=vars(self.args))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.9)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for mfcc_seq, wavnet_seq, labels in self.train_loader:
                mfcc_seq, wavnet_seq, labels = mfcc_seq.to(self.device), wavnet_seq.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(mfcc_seq, wavnet_seq)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * mfcc_seq.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            train_loss = running_loss / total
            train_acc = correct / total
            val_loss, val_acc = self.evaluate()
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        torch.save(self.model.state_dict(), 'rnn_features_model.pt')
        wandb.finish()
    def evaluate(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for mfcc_seq, wavnet_seq, labels in self.test_loader:
                mfcc_seq, wavnet_seq, labels = mfcc_seq.to(self.device), wavnet_seq.to(self.device), labels.to(self.device)
                outputs = self.model(mfcc_seq, wavnet_seq)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * mfcc_seq.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += sum(predicted == labels).item()
                total += labels.size(0)
        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc
    def save(self, path):
        torch.save(self.model.state_dict(), path)
