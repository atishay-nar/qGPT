import argparse
import yaml
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
from reduced_mnist import ReducedMNISTDataset
from qGPT import QuantumImageGPT
from utils import quantize, unquantize

def generate(model, context, length, num_samples=1):
    output = context.unsqueeze(0).repeat(num_samples, 1)  # add batch size of num_samples so shape [seq len, batch]

    # predict
    with torch.no_grad():
        for _ in tqdm(range(length), leave=False, desc=f"Generating"):
            logits = model(output)               # (batch, seq, vocab)
            logits = logits[:, -1, :]            # (batch, vocab)
            probs = F.softmax(logits, dim=-1)  # convert logits to probabilities
            pred = torch.multinomial(probs, num_samples=1) # sample from the distribution
            output = torch.cat((output, pred), dim=1)  # append the predicted token to the output

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_num", required=True, type=int)
    parser.add_argument("--n_examples", default=5, type=int)
    parser.add_argument("--n_samples", default=5, type=int)
    args, _ = parser.parse_known_args()
    cfg = argparse.Namespace(**yaml.safe_load(open("configs.yml", "r")))
