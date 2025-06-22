import pennylane as qml
from pennylane.math import fidelity
import numpy as np
import torch
import torch.nn as nn
import argparse
import yaml
from embeddings import zz_feature_map

# embed vector and get mixed state
def get_mixed_state(x):
    wires = list(range(len(x)))
    qml.IQPEmbedding(x, wires=wires)
    return qml.density_matrix(wires=wires)
    

# single-head quantum mixed-state self-attention
def single_qmsan(Q, K, V, dev):
     # batch size and number of tokens per attention head
    B, S, E = Q.shape
    
    # create blank tensor to hold attention values
    attn = np.zeros((B, S, S)) 

    # compute dot product of each query key pair in quantum
    for b in range(B):
        for i in range(S):
            for j in range(S):
                q = Q[b][i]
                k = K[b][j]
                # get mixed states
                rho_q = qml.QNode(get_mixed_state, dev)(q)
                rho_k = qml.QNode(get_mixed_state, dev)(k)
                # get fidelity score
                score = fidelity(rho_q, rho_k)
                attn[b][i][j] = score
    
    # scale attention scores to range [-1, 1]
    attn = 2 * attn - np.ones_like(attn)
    # to do: mask to avoid peaking at future tokens. before or after scaling?
    return attn @ V

# torch module of single-head quantum mixed-state self-attention
class SingleQMSANHead(nn.Module):
    def __init__(self, embed_dim, head_dim, dev):
        super().__init__()
        
        self.head_dim =  head_dim

        # embeds Q, K, V into quantum states and returns their mixed states
        self.circuit = qml.QNode(get_mixed_state, dev, interface="torch") 

        # classical linear projection for V since we matmul post-measurement
        self.V_linear = nn.Linear(embed_dim, self.head_dim, bias=False)

    def forward(self, Q, K, V): # TO DO: change this, this sucks. figure out how to embed Q, K, V for an x vector

        # V = self.V_linear(x)
        
        # batch size and number of tokens per attention head
        B, S, E = Q.shape
        
        # create blank tensor to hold attention values
        attn = torch.zeros((B, S, S), dtype=torch.float32) 

        # compute dot product of each query key pair in quantum
        for b in range(B):
            for i in range(S):
                for j in range(S):
                    q = Q[b][i]
                    k = K[b][j]
                    # get mixed states
                    rho_q = self.circuit(q)
                    rho_k = self.circuit(k)
                    # get fidelity score
                    score = fidelity(rho_q, rho_k)
                    attn[b][i][j] = score
        
        # scale attention scores to range [-1, 1]
        attn = 2 * attn - torch.ones_like(attn, dtype=torch.float32)

        out = torch.matmul(attn, V)

        # TO DO: mask to avoid attending on future tokens
        return out
    
# multi-headed QMSAN
class MultiHeadQMSAN(nn.Module):
    def __init__(self, embed_dim, n_heads, dev):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embed dimension must be divisible by number of heads"
        self.head_size = embed_dim // n_heads

        # final linear projection for MHA
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # single-head QMSAN heads
        self.heads = nn.ModuleList([SingleQMSANHead(embed_dim, self.head_size, dev) for _ in range(n_heads)])

    def forward(self, x):
        # concatenate all heads
        out = torch.cat([head(x)for head in self.heads], dim=-1)

        # linear proj
        out = self.W_o(out)
        return out

if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)

    Q = np.array([[[2, 1], [1, 2]],[[2, 1], [1, 2]]])
    K = np.array([[[1, -2], [1, 0]],[[0, 1], [1, 2]]])
    V = np.array([[[1, 0], [0, 1]],[[1, 0], [0, 1]]])

    dev = qml.device("default.mixed", wires=(Q.shape[-1]))
    print(single_qmsan(Q, K, V, dev))

    # test torch module
    module = SingleQMSANHead(cfg.EMBED_DIM, cfg.NUM_HEADS, dev)
    print(module(
                torch.tensor(Q, dtype=torch.float32), 
                torch.tensor(K, dtype=torch.float32), 
                torch.tensor(V, dtype=torch.float32)
                ))
   
