# imports
import argparse
import yaml
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from reduced_mnist import ReducedMNISTDataset
from utils import quantize

# set device
DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
        )
        
# set random seed
np.random.seed(37)

# class that transforms a dataset, here MNIST, into a tokenized array based on the centroids
class TokenizedData(Dataset):
    
    def __init__(self, tokenized_data_dir, num_clusters, img_size, classes, samples_per_class):
        super().__init__()

        # Where to cache tokenized data
        os.makedirs(tokenized_data_dir, exist_ok=True)
        self.cache_path = os.path.join(tokenized_data_dir, f"MNIST_tokens_{num_clusters}.npy")

        # If cache exists, load directly; otherwise quantize all images and save
        if os.path.exists(self.cache_path):
            self.tokenized = np.load(self.cache_path, mmap_mode="r").copy()
        else:
            self.SEQ_LEN = img_size * img_size
            # Load raw MNIST dataset 
            reduced = ReducedMNISTDataset(img_size, classes=classes, samples_per_class=samples_per_class)

            # get centroids
            centroids_path = os.path.join(cfg.CENTROID_DIR, f"centroids_{num_clusters}.npy")
            self.centroids = torch.tensor(np.load(centroids_path)).to(DEVICE)
            self.tokenized = self.quantize_and_cache(reduced, self.centroids)
    
    # quantize images
    def quantize_and_cache(self, pre_images, centroids):
        # create empty array to hold quantized images
        token_array = np.empty((len(pre_images), self.SEQ_LEN), dtype=np.int32)
        # loop through images
        i = 0
        for img in tqdm(pre_images, desc="Quantizing Images"):

            img = img.to(DEVICE) # create image tensor 
            tokens = quantize(img, centroids) # quantize image to tokens
            token_array[i] = tokens.cpu().numpy() # store in array
            i += 1

        # save tokenized data to cache
        np.save(self.cache_path, token_array)
        print(f"Saved tokenized data to {self.cache_path}")
        return token_array
    
    # needed for Dataset class
    def __len__(self):
        return len(self.tokenized)
    
    # needed for Dataset class
    def __getitem__(self, idx):
        tokens = self.tokenized[idx]
        return torch.from_numpy(tokens)

            
if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)
    test = TokenizedData(cfg.TOKENIZED_DATA_DIR, cfg.NUM_CLUSTERS, cfg.IMAGE_SIZE, classes=cfg.CLASSES, samples_per_class=cfg.SAMPLES_PER_CLASS)


    