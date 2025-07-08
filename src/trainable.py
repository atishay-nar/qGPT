import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import yaml
import time
from embeddings import circuit_14
from utils import js_divergence

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


# embed vector and probs
def circuit(inputs, weights):
    n = int(np.log2(len(inputs)))
    reps = len(weights) // 4
    wires = list(range(n))
    qml.AmplitudeEmbedding(inputs, wires=wires, normalize=True)
    circuit_14(weights, wires=wires, reps=reps)
    return qml.probs()


# torch module of single-head quantum mixed-state self-attention
class SingleQMSANHead(nn.Module):
    def __init__(self, embed_dim, head_dim, dev, n_qlayers):
        super().__init__()

        self.head_dim = head_dim
        n_qubits = int(np.log2(embed_dim))

        # classical linear projections only for V
        self.V_proj = nn.Linear(embed_dim, self.head_dim, bias=False)

        # embeds Q and K into quantum states and returns their mixed states
        self.circuit = qml.QNode(circuit, dev, interface="torch")

        # weight shapes for Torch Layer
        self.weight_shapes = {"weights": (4 * n_qlayers, n_qubits)}

        # trainable quantum layers for Q and K
        self.Q_layer = qml.qnn.TorchLayer(
            self.circuit, weight_shapes=self.weight_shapes
        )
        self.K_layer = qml.qnn.TorchLayer(
            self.circuit, weight_shapes=self.weight_shapes
        )


    def forward(self, x): 

        V = self.V_proj(x)

        # batch size and number of tokens per attention head
        B, S, E = V.shape

        # create blank tensor to hold attention values
        attn = torch.zeros((B, S, S), dtype=torch.float32, device=x.device)

        
        # compute dot product of each query key pair in quantum
        for b in range(B):
            # get mixed states for each qi and kj and cache for efficiency
            # certainly not possible with hardware
            rho_Q = []
            rho_K = []
            for i in range(S):
                rho_Q.append(self.Q_layer(x[b][i]))
                rho_K.append(self.K_layer(x[b][i]))

            for i in range(S):
                for j in range(S):
                    if j > i:  # mask
                        score = 0.0
                    else:
                        # get overlap
                        score = 1 - js_divergence(rho_Q[i], rho_K[j])

                    attn[b][i][j] = score
                        
        # normalize
        attn = F.normalize(attn, p=1, dim=-1)
        # scale attention scores to range [-1, 1] ASK GLEN
        # attn = 2 * attn - torch.ones_like(attn, dtype=torch.float32)

        out = torch.matmul(attn, V)
        return out


# multi-headed QMSAN
class MultiHeadQMSAN(nn.Module):
    def __init__(self, embed_dim, n_heads, dev, n_qlayers):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embed dimension must be divisible by number of heads"
        self.head_size = embed_dim // n_heads

        # final linear projection for MHA
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # single-head QMSAN heads
        self.heads = nn.ModuleList(
            [SingleQMSANHead(embed_dim, self.head_size, dev, n_qlayers) for _ in range(n_heads)]
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

    x = torch.ones((1, 100, 16), dtype=torch.float32).to(DEVICE)

    # test torch module
    if torch.cuda.is_available():
        dev = qml.device("lightning.qubit", wires=4)
    else:
        dev = qml.device("default.qubit", wires=4)
    start = time.time()
    module = MultiHeadQMSAN(16, 4, dev, 1).to(DEVICE)
    end = time.time()
    print(module(x))
    end = time.time()
    print(f"Time taken to run module: {end - start:.4f} seconds")
