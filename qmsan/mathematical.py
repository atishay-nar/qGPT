import pennylane as qml
from pennylane.math import fidelity
import numpy as np
import torch
import argparse
import yaml
from embeddings import zz_feature_map

# embed vector and get mixed state
def get_mixed_state(x):
    wires = list(range(len(x)))
    zz_feature_map(x, wires=wires, reps=1)
    return qml.density_matrix(wires=wires)
    

# single-head quantum mixed-state self-attention
def single_qmsan(Q, K, V, dev):
     # batch size and number of tokens per attention head
    B, N, _ = Q.shape
    
    
    # create blank tensor to hold attention values
    attn = np.zeros((B, N, N)) 

    # compute dot product of each query key pair in quantum
    for b in range(B):
        for i in range(N):
            for j in range(N):
                q = Q[b][i]
                k = K[b][j]
                # get mixed states
                rho_q = qml.QNode(get_mixed_state, dev)(q)
                rho_k = qml.QNode(get_mixed_state, dev)(k)
                # get fidelity score
                score = fidelity(rho_q, rho_k)
                attn[b][i][j] = 2 * score - 1 # scale to [-1, 1]
    return attn


    


if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)

    # if cfg.EMBED_DIM % cfg.NUM_HEADS != 0:
    #     raise ValueError("EMBED_DIM must be divisible by NUM_HEADS")
    # # dim_attention = int(cfg.EMBED_DIM / cfg.NUM_HEADS)

    Q = np.array([[[2, 1], [1, 2]],[[2, 1], [1, 2]]])
    K = np.array([[[1, 2], [1, 0]],[[0, 1], [1, 2]]])
    V = np.array([[[1, 0], [0, 1]],[[1, 0], [0, 1]]])

    dev = qml.device("default.mixed", wires=(Q.shape[-1]))
    print(single_qmsan(Q, K, V, dev))
   
