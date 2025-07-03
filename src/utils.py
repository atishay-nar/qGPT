import torch
import argparse
import yaml
import pennylane as qml

# function to compute squared euclidian distance
def squared_euclidean_distance(a, b):
    b = torch.transpose(b, 0, 1)
    a2 = torch.sum(torch.square(a), dim=1, keepdim=True)
    b2 = torch.sum(torch.square(b), dim=0, keepdim=True)
    ab = torch.matmul(a, b)
    d = a2 - 2 * ab + b2
    return d

# function to quantize a single image as a string of tokens based on centroids
def quantize(img, centroids):
    img = img.permute(1, 2, 0).contiguous()
    img = img.view(-1, 1) # flatten to pixels

    # calc distance to centroids
    d = squared_euclidean_distance(img, centroids)
    tokens = torch.argmin(d, 1) # choose closest centroid index for each pixel
    return tokens

# unquantize tokens to pixel values using centroids
def unquantize(tokens, centroids):
    return centroids[tokens]


# parameter count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    cfg = argparse.Namespace(**yaml.safe_load(open("configs.yml", "r")))

    import pennylane as qml
    from qGPT import QuantumImageGPT

    num_qubits = cfg.EMBED_DIM / cfg.NUM_HEADS
    dev = qml.device("default.mixed", wires=int(num_qubits))
    model = QuantumImageGPT(
        vocab_size=cfg.NUM_CLUSTERS,
        embed_dim=cfg.EMBED_DIM,
        n_heads=cfg.NUM_HEADS,
        n_layers=cfg.NUM_LAYERS,
        image_size=cfg.IMAGE_SIZE,
        quantum_device=dev,
        n_qlayers=cfg.CIRCUIT_REPS
    )
    
    print(f"Model has {count_parameters(model):,} trainable parameters.")