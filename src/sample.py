import argparse
import yaml
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pennylane as qml
from PIL import Image
from reduced_mnist import ReducedMNISTDataset
from qGPT import QuantumImageGPT
from utils import quantize, unquantize
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# set random seed 
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

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

def sample(model, checkpoint_num, device, centroid_dir, num_clusters, test_data, img_size, classes, n_examples=3, n_samples=3, specific_ckpt=None):
    os.makedirs('./figures', exist_ok=True)

    # load model checkpoint
    if specific_ckpt:
        ckpt = torch.load(specific_ckpt, map_location=device)
    else:
        ckpt = torch.load(f"./checkpoints/qgpt_epoch_{checkpoint_num}.pth", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # load centroids,
    centroids_path = os.path.join(centroid_dir, f"centroids_{num_clusters}.npy")
    centroids = torch.tensor(np.load(centroids_path)).to(device)

    loader = iter(DataLoader(test_data, batch_size=1, shuffle=True))


    rows=[]
    visited = set()
    ssim = []
    psnr = []
    for example in tqdm(range(n_examples), desc="Sampling Images"):
        # get random image
        img, label = next(loader)

        # ensure no duplicate classes before all classes are visited
        if len(visited) < len(classes):
            while label.item() in visited:
                img, label = next(loader)
            visited.add(label.item())

        img = img[0].to(device)

        # quantize image to tokens
        img = quantize(img, centroids).cpu().numpy() # get tokens and flatten into seq
        tokens = img.reshape(-1)
        img = img.reshape(img_size, img_size) # for plotting

        # choose context. here we use the first half of the image
        context = tokens[:int(len(tokens) / 2)]
        context_img = np.pad(context, (0, int(len(tokens) / 2))).reshape(img_size, img_size) # for plotting
        context = torch.from_numpy(context).long().to(device)  # convert to tensor and move to device

        # generate the rest of the image from the context
        pred = generate(model, context, int(len(tokens) / 2), num_samples=n_samples).cpu().numpy()
        pred = pred.reshape(-1, img_size, img_size)  # reshape to image size

        # evaluate the generated images
        for pred_img in pred:
            # unquantize the truth and predicted
            gen = unquantize(pred_img, centroids).squeeze(-1).cpu().numpy() # squeeze because we do not need channels
            truth = unquantize(img, centroids).squeeze(-1).cpu().numpy()
            
            data_range = max(gen.max(), truth.max()) - min(gen.min(), truth.min()) # range 

            # calculate SSIM and PSNR
            ssim.append(structural_similarity(truth, gen, data_range=data_range))
            psnr.append(peak_signal_noise_ratio(truth, gen, data_range=data_range))

        # add example to rows
        rows.append(np.concatenate([context_img[None, ...], pred, img[None, ...]], axis=0)) # 

    # stack all rows together
    fig = np.stack(rows, axis=0)  

    nrow, ncol, h, w = fig.shape
    fig = unquantize(fig.swapaxes(1, 2).reshape(h * nrow, w * ncol), centroids).cpu().numpy()
    fig = (fig * 255).round().astype(np.uint8)
    pic = Image.fromarray(np.squeeze(fig))
    if specific_ckpt:
        pic.save(f"./figures/sample_from_{specific_ckpt.split('/')[-1].split('.')[0]}.png")
    else:
        pic.save(f"./figures/sample_at_epoch_{checkpoint_num}.png")

    print(f'''
    SSIM: mean {np.mean(ssim):.4f} std {np.std(ssim):.4f} high {np.max(ssim):.4f} low {np.min(ssim):.4f}
    PSNR: mean {np.mean(psnr):.4f} std {np.std(psnr):.4f} high {np.max(psnr):.4f} low {np.min(psnr):.4f}
''')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--specific", default=False, type=bool)
    parser.add_argument("--n_examples", default=5, type=int)
    parser.add_argument("--n_samples", default=5, type=int)
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

    if args.specific:
        specific_ckpt = args.ckpt
        ckpt = 0
    else:
        specific_ckpt = None
        ckpt = args.ckpt

    sample(
        model=model,
        checkpoint_num=ckpt,
        device=DEVICE,
        centroid_dir=cfg.CENTROID_DIR,
        num_clusters=cfg.NUM_CLUSTERS,
        test_data=test_data,
        img_size=cfg.IMAGE_SIZE,
        classes=cfg.CLASSES,
        specific_ckpt=specific_ckpt
    )
