#imports
import torch.nn as nn
from qmsan.mathematical_nograd import MultiHeadQMSAN

# quantum-clssical hybrid transformer model that utilizes QMSAN for attention
class HybridTransformer(nn.Module):
    def __init__(self, embed_dim, n_heads, quantum_device):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        # Normalization layers
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        #multi-headed QMSAN attention layer
        self.multi_head_qmsan = MultiHeadQMSAN(embed_dim, n_heads, dev=quantum_device)

        # multi-layer perceptron for feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        # normalize
        x = self.ln1(x)

        # multi-headed QMSAN attention
        x = x + self.multi_head_qmsan(x)

        # normalize
        x = self.ln2(x)
        
        # feed-forward
        out = x + self.mlp(x)
        return out

