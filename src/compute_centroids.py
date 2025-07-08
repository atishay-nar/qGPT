# imports
import argparse
import yaml
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from reduced_mnist import ReducedMNISTDataset

        
# set random seed
np.random.seed(37)

# centroid calculation function
def compute_centroids(num_clusters, img_size, classes, samples_per_class, centroid_dir):

    # load MNIST train data (only pixel arrays, no labels)
    reduced_mnist = ReducedMNISTDataset(img_size, classes=classes, samples_per_class=samples_per_class) 

    # transform into pixel array
    train_x = np.stack([x.numpy()for x in reduced_mnist])
    train_x = train_x.transpose(0, 2, 3, 1)
    
    pixels = train_x.reshape(-1, train_x.shape[-1])

    # perform k-means clustering
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=8192, verbose=1).fit(pixels)
    centroids = kmeans.cluster_centers_  # shape (num_clusters, 1)

    # Save centroids to file 
    # SOL cannot use SKLearn efficiently so it is more effective to precompute centroids on local machine
    centroid_path = os.path.join(centroid_dir, f"centroids_{num_clusters}.npy")
    os.makedirs(centroid_dir, exist_ok=True)
    np.save(centroid_path, centroids)
    print(f"Saved centroids to {centroid_path}")

# main
if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)
    centroid = compute_centroids(img_size=cfg.IMAGE_SIZE,
                      classes=cfg.CLASSES, 
                      samples_per_class=cfg.SAMPLES_PER_CLASS, 
                      centroid_dir=cfg.CENTROID_DIR, 
                      num_clusters=cfg.NUM_CLUSTERS
                      )