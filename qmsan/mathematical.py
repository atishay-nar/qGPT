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
    
    # scale attnetion scores to range [-1, 1]
    attn = 2 * attn - np.ones_like(attn)
    # to do: mask to avoid peaking at future tokens
    return attn @ V

# torch module of single-head quantum mixed-state self-attention
class SingleQMSANHead(nn.Module):
    def __init__(self, dev):
        super().__init__()

        # embeds Q, K, V into quantum states and returns their mixed states
        self.circuit = qml.QNode(get_mixed_state, dev, interface="torch") 

    def forward(self, Q, K, V): # TO DO: change this, this sucks. figure out how to embed Q, K, V for an x vector
        
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
        
        # scale attnetion scores to range [-1, 1]
        attn = 2 * attn - torch.ones_like(attn, dtype=torch.float32)

        # TO DO: mask to avoid atteneding on future tokens
        return torch.matmul(attn, V)

    

if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)

    Q = np.array([[[2, 1], [1, 2]],[[2, 1], [1, 2]]])
    K = np.array([[[1, -2], [1, 0]],[[0, 1], [1, 2]]])
    V = np.array([[[1, 0], [0, 1]],[[1, 0], [0, 1]]])

    dev = qml.device("default.mixed", wires=(Q.shape[-1]))
    print(single_qmsan(Q, K, V, dev))

    # test torch module
    module = SingleQMSANHead(dev)
    print(module(
                torch.tensor(Q, dtype=torch.float32), 
                torch.tensor(K, dtype=torch.float32), 
                torch.tensor(V, dtype=torch.float32)
                ))
   
