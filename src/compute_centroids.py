# imports
import argparse
import yaml
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from sklearn.cluster import MiniBatchKMeans

        
# set random seed
np.random.seed(0)

# centroid calculation function
def compute_centroids(cfg):
    num_clusters = cfg.NUM_CLUSTERS
    img_size = cfg.IMAGE_SIZE

    # transform to convert images to tensors and resize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size))
    ])

    # load MNIST train data (only pixel arrays, no labels)
    mnist = torchvision.datasets.MNIST(
        root=cfg.DATA_DIR, train=True, download=True,
        transform=transform
    )

    # transform into pixel array
    train_x = np.stack([x.numpy()for x, _ in mnist])
    train_x = train_x.transpose(0, 2, 3, 1)
    
    pixels = train_x.reshape(-1, train_x.shape[-1])

    # perform k-means clustering
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=8192, verbose=1).fit(pixels)
    centroids = kmeans.cluster_centers_  # shape (num_clusters, 3)

    # Save centroids to file
    centroid_path = os.path.join(cfg.CENTROID_DIR, f"centroids_{num_clusters}.npy")
    os.makedirs(cfg.CENTROID_DIR, exist_ok=True)
    np.save(centroid_path, centroids)
    print(f"Saved centroids to {centroid_path}")

# main
if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)
    compute_centroids(cfg)