import argparse
import yaml
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate():
    return

if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)
    print(evaluate())

