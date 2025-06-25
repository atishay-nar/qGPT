# imports
import argparse
import yaml
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import DataLoader
from qGPT import QuantumImageGPT 
from tokenize_data import TokenizedData

# train for quantum image GPT
def train(train_data, model_save_dir, batch_size, num_workers, model, lr, epochs, save_interval, weight_decay=None, use_scheduler=False):
    os.makedirs(model_save_dir, exist_ok=True)

    # load data
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    # set optimizer
    if weight_decay:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # optimize learning rate scheduler
    if use_scheduler:
        # cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader)) 

    # training loop
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch = batch.long().to(DEVICE) # batch: (B, seq_len)
            inputs = batch[:, :-1] # first seq_len-1 tokens for each image
            targets = batch[:, 1:] # last seq_len-1 tokens for each image

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) # calculate loss based on logits and targets
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # optional scheduler step
        if use_scheduler:
            scheduler.step()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        # Save checkpoint periodically
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            save_path = os.path.join(model_save_dir, f"image_gpt_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)

    # set device
    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
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
        quantum_device=dev
    ).to(DEVICE)

    train_data = TokenizedData(
        tokenized_data_dir=cfg.TOKENIZED_DATA_DIR,
        centroid_dir=cfg.CENTROID_DIR,
        num_clusters=cfg.NUM_CLUSTERS,
        img_size=cfg.IMAGE_SIZE,
        classes=cfg.CLASSES,
        samples_per_class=cfg.SAMPLES_PER_CLASS
    )

    train(
        train_data=train_data,
        model_save_dir=cfg.MODEL_SAVE_DIR,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        model=model,
        lr=cfg.LR,
        epochs=cfg.EPOCHS,
        save_interval=cfg.SAVE_INTERVAL,
        weight_decay=cfg.WEIGHT_DECAY
        )