import pennylane as qml
from pennylane.math import fidelity
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import yaml
from tqdm import tqdm
import time
# from embeddings import zz_feature_map

# set device
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps" if torch.mps.is_available()
    else "cpu"
)

# set random seed
np.random.seed(37)
torch.manual_seed(37)


# embed vector and get mixed state
# weights not needed for this example, but can be used for more complex embeddings
def get_mixed_state(inputs, weights):
    n =len(inputs)
    wires = list(range(n))
    qml.IQPEmbedding(inputs, wires=wires)
    return qml.density_matrix(wires=wires[:n//2])


# torch module of single-head quantum mixed-state self-attention
class SingleQMSANHead(nn.Module):
    def __init__(self, embed_dim, head_dim, dev):
        super().__init__()

        self.head_dim = head_dim

        # classical linear projections to get to from embed_dim to head_dim
        self.Q_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.K_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.V_proj = nn.Linear(embed_dim, self.head_dim, bias=False)

        # embeds Q and K into quantum states and returns their mixed states
        self.circuit = qml.QNode(get_mixed_state, dev, interface="torch")

        # weight shapes for Torch Layer
        self.weight_shapes = {"weights": head_dim}

        # trainable quantum layers for Q and K
        self.Q_mixed = qml.qnn.TorchLayer(
            self.circuit, weight_shapes=self.weight_shapes
        )
        self.K_mixed = qml.qnn.TorchLayer(
            self.circuit, weight_shapes=self.weight_shapes
        )
    
    

    def forward(self, x): 

        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)

        # batch size and number of tokens per attention head
        B, S, E = Q.shape

        # create blank tensor to hold attention values
        attn = torch.zeros((B, S, S), dtype=torch.float32, device=x.device)

        
        # compute dot product of each query key pair in quantum
        for b in range(B):
            # get mixed states for each qi and kj and cache for efficiency
            # certainly not possible with hardware
            rho_Q = []
            rho_K = []
            for i in tqdm(range(S)):
                rho_Q.append(self.Q_mixed(Q[b][i]))
                rho_K.append(self.K_mixed(K[b][i]))               

            for i in tqdm(range(S)):
                for j in range(S):
                    if j > i:  # mask
                        score = 0.0
                    else:
                        # get overlap
                        score = torch.trace(torch.matmul(rho_Q[i], rho_K[j]))

                    attn[b][i][j] = score
        # normalize
        attn = F.normalize(attn, p=1, dim=-1)
        # scale attention scores to range [-1, 1]
        #attn = 2 * attn - torch.ones_like(attn, dtype=torch.float32)

        out = torch.matmul(attn, V)
        return out


# multi-headed QMSAN
class MultiHeadQMSAN(nn.Module):
    def __init__(self, embed_dim, n_heads, dev):
        super().__init__()
        assert (
            embed_dim % n_heads == 0
        ), "Embed dimension must be divisible by number of heads"
        self.head_size = embed_dim // n_heads

        # final linear projection for MHA
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # single-head QMSAN heads
        self.heads = nn.ModuleList(
            [SingleQMSANHead(embed_dim, self.head_size, dev) for _ in range(n_heads)]
        )

    def forward(self, x):
        # concatenate all heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        # linear proj
        out = self.W_o(out)
        return out


if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)

    Q = np.array([[[2, 1], [1, 2]], [[2, 1], [1, 2]]])
    K = np.array([[[1, -2], [1, 0]], [[0, 1], [1, 2]]])
    V = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    x = torch.ones((1, 100, 16), dtype=torch.float32).to(DEVICE)

    # test torch module
    if torch.cuda.is_available():
        dev = qml.device("lightning-gpu", wires=8, gpu=True)
    else:
        dev = qml.device("default.mixed", wires=8)
    start = time.time()
    module = MultiHeadQMSAN(16, 2, dev).to(DEVICE)
    end = time.time()
    print(module(x))
    end = time.time()
    print(f"Time taken to run module: {end - start:.4f} seconds")
