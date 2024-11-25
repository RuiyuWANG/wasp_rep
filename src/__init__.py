import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lightning.pytorch as pl

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.classifier = nn.Linear(hidden_dim, 10) # MNIST has 10 classes (0-9)

    def forward(self, x, class_out=False):
        if not class_out:
            return self.decoder(torch.relu(self.encoder(x)))
        else:
            return self.classifier(torch.relu(self.encoder(x)))

class NoiseDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size=128):
        super().__init__()
        self.data_dir = path
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        datasets.MNIST(root=self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        # train_dataset = datasets.MNIST(root=self.data_dir, train=True, transform=self.transform)
        # self.train_set, self.val_set = torch.utils.data.random_split(train_dataset, [55000, 5000])
        # same to create_initial_Xs_distill()
        self.train_set = torch.rand(28*28, 20*15)
        self.train_set /= 16

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size=128):
        super().__init__()
        self.data_dir = path
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        train_dataset = datasets.MNIST(root=self.data_dir, train=True, transform=self.transform)
        self.train_set, self.val_set = torch.utils.data.random_split(train_dataset, [55000, 5000])
        self.test_set = datasets.MNIST(root=self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=16,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
