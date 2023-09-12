import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision as tv
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class RealFakeDataset(Dataset):
    def __init__(self, path_real, path_fake, split="train", transform=None):

        self.path_real = path_real
        self.path_fake = path_fake
    
        self.real_images = sorted(os.listdir(self.path_real))
        self.fake_images = sorted(os.listdir(self.path_fake))
        self.real_images = [os.path.join(self.path_real, img) for img in self.real_images]
        self.fake_images = [os.path.join(self.path_fake, img) for img in self.fake_images]

        self.transform = transform

        if split == "train":
            self.real_images = self.real_images[:int(0.8*len(self.real_images))]
            self.fake_images = self.fake_images[:int(0.8*len(self.fake_images))]
        elif split == "val":
            self.real_images = self.real_images[int(0.8*len(self.real_images)):]
            self.fake_images = self.fake_images[int(0.8*len(self.fake_images)):]

        assert len(self.real_images) == len(self.fake_images)

        # Define an offset in indices for the fake images.
        # The offset is reset every epoch to change the pairing
        # while making each image appear once per epoch.
        self.random_modulo = np.random.randint(0, len(self.real_images))

        self._cache = {}

    def __len__(self):
        return len(self.real_images)
    
    def __getitem__(self, idx):
        real = self._cache_image_loader(self.real_images[idx])
        fake = self._cache_image_loader(self.fake_images[idx % self.random_modulo]) 

        if self.transform:
            real = self.transform(real)
            fake = self.transform(fake)

        return real, fake

    def _cache_image_loader(self, path):
        if not path in self._cache:
            self._cache[path] = Image.open(path)
        return self._cache[path]

def get_dataloaders(config):
    train_ds = RealFakeDataset(
        path_real=config.data.path_real,
        path_fake=config.data.path_fake,
        split="train",
        transform=tv.transforms.ToTensor()
    )
    val_ds = RealFakeDataset(
        path_real=config.data.path_real,
        path_fake=config.data.path_fake,
        split="val",
        transform=tv.transforms.ToTensor()
    )

    if config.data.num_workers == "auto":
        config.data.num_workers = min(config.data.batch_size, os.cpu_count()//torch.cuda.device_count())

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    return train_dl, val_dl
