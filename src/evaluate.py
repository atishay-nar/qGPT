import argparse
import yaml
import torch
import pennylane as qml
from qGPT import QuantumImageGPT
from reduced_mnist import ReducedMNISTDataset
from utils import quantize, unquantize
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate(model, checkpoint, device, centroid_dir, num_clusters, test_data, img_size, n_examples=3, n_samples=3):
    return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name", required=True, type=str)
    args, _ = parser.parse_known_args()
    cfg = argparse.Namespace(**yaml.safe_load(open("configs.yml", "r")))

    # set device
    DEVICE = (
        "cuda" if torch.cuda.is_available()
        # else "mps" if torch.mps.is_available()
        else "cpu"
        )
    
    # set quantum device
    num_qubits = int(cfg.EMBED_DIM / cfg.NUM_HEADS)

    if torch.cuda.is_available():
        dev = qml.device("lightning.gpu", wires=num_qubits)
    else:
        dev = qml.device("default.mixed", wires=num_qubits)

    # define model
    model = QuantumImageGPT(
        vocab_size=cfg.NUM_CLUSTERS,
        embed_dim=cfg.EMBED_DIM,
        n_heads=cfg.NUM_HEADS,
        n_layers=cfg.NUM_LAYERS,
        image_size=cfg.IMAGE_SIZE,
        quantum_device=dev,
        n_qlayers=cfg.CIRCUIT_REPS
    ).to(DEVICE)

    test_data = ReducedMNISTDataset(cfg.IMAGE_SIZE, classes=cfg.CLASSES, samples_per_class=cfg.SAMPLES_PER_CLASS, train=False)

