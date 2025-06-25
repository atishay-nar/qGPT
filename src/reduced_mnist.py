import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

# reducing size of images, classes, and number of samples per class
class ReducedMNISTDataset(Dataset):
    def __init__(self, img_size, classes, samples_per_class, raw_data_dir="./data"):
        super().__init__()
        #
        amts = {}
        for c in classes:
            amts[c] = samples_per_class

        transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # <-- resize
        transforms.ToTensor(),                    # <-- to tensor
        ])

        # load raw data
        raw_dataset = torchvision.datasets.MNIST(root=raw_data_dir, train=True, download=True, transform=transform)
        self.reduced = self.reduce_data(raw_dataset, amts)
    
    def reduce_data(self, raw_dataset, amts):
        reduced = []
        # filter
        for itm in raw_dataset:
            img, label = itm
            if label in amts and amts[label] > 0:
                reduced.append(img)
                amts[label] -= 1
    
        return reduced

    # needed for Dataset class
    def __len__(self):
        return len(self.reduced)
    
    # needed for Dataset class
    def __getitem__(self, idx):
        img = self.reduced[idx]
        return img

if __name__ == "__main__":
    cfg = argparse.Namespace(**yaml.safe_load(open("configs.yml", "r")))
    test = ReducedMNISTDataset(cfg.IMAGE_SIZE, classes=cfg.CLASSES, samples_per_class=cfg.SAMPLES_PER_CLASS)
    print(test[0])