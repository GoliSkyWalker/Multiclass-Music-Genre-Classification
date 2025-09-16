
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import wandb
import random
from utils.csv_utils import read_aggregated_features



class TorchNNModel:
    def __init__(self, args):
        self.args = args
        seed = getattr(args, 'seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model().to(self.device)
        self.epochs = getattr(args, 'nn_epochs', 1000)
        self.batch_size = getattr(args, 'nn_batch_size', 8)
        self.lr = getattr(args, 'nn_lr', 1e-5)
        self.load_data()

    def load_data(self):
        df = read_aggregated_features(self.args)

        # Select features based on args.features
        if self.args.features == 'base':
            # Use only MFCC features
            df['mfcc'] = df['mfcc'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
            features = np.stack(df['mfcc'].values)
        else:
            # Use both Wavenet and MFCC features (concatenate)
            df['wavnet'] = df['wavnet'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
            df['mfcc'] = df['mfcc'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
            features = np.concatenate([np.stack(df['wavnet'].values), np.stack(df['mfcc'].values)], axis=1)

        classes = df['class'].values
        X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.15, random_state=42, shuffle=True)
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        self.n_classes = len(le.classes_)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        self.train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.batch_size)

    def build_model(self):
        return nn.Sequential(
            nn.Linear(39, 512),
            nn.Tanh(),
            nn.Linear(512, 100),
            nn.Tanh(),
            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )

    def train(self):
        wandb.init(project=self.args.wandb_project, config=vars(self.args))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.9)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
                train_loss = running_loss / total
                train_acc = correct / total
            val_loss, val_acc = self.evaluate()
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        torch.save(self.model.state_dict(), 'rms2acc46_torch.pt')
        wandb.finish()

    def evaluate(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def save(self, path):
        torch.save(self.model.state_dict(), path)
