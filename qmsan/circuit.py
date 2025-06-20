import pennylane as qml
import numpy as np
import torch
import argparse
import yaml
from embeddings import zz_feature_map

# swap test to get similarity score for two quantum states
def swap_test(q_vector, k_vector, q_wires, k_wires):

    # embed
    zz_feature_map(q_vector, q_wires)
    zz_feature_map(k_vector, k_wires)

    # apply Hadamard on ancilla wire
    qml.Hadamard(wires=0)

    # controlled-SWAP on pairs
    for i in range(len(q_wires)): 
        qml.CSWAP(wires=[0, q_wires[i], k_wires[i]])

    # final Hadamard on ancilla wire
    qml.Hadamard(wires=0)

    # return probability of ancilla wire being in |0> state
    return qml.probs(wires=0)


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

                q_wires = list(range(1, 1 + N))
                k_wires = list(range(1 + N, 1 + 2 * N))

                # get score with swap test
                score = qml.QNode(swap_test, dev)(q, k, q_wires, k_wires)
                attn[b][i][j] = 2*score[0] - 1 # scale to [-1, 1]
    return attn @ V
    
   

    


if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)

    if cfg.EMBED_DIM % cfg.NUM_HEADS != 0:
        raise ValueError("EMBED_DIM must be divisible by NUM_HEADS")
    dim_attention = int(cfg.EMBED_DIM / cfg.NUM_HEADS)

    # define device
    dev = qml.device("default.qubit", wires=(1 + 2 * dim_attention))
    Q = np.array([[[2, 1], [1, 2]],[[2, 1], [1, 2]]])
    K = np.array([[[2, 1], [1, 2]],[[2, 0], [1, 2]]])
    V = np.array([[[1, 0], [0, 1]],[[1, 0], [0, 1]]])
    print(single_qmsan(Q, K, V, dev))