#imports
import torch
import torch.nn as nn
from hybrid_transformer import HybridTransformer

# Image GPT model with a quantum-classical hybrid transformer
class QuantumImageGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, image_size, quantum_device):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.image_size = image_size
        self.max_seq_len = image_size * image_size
        self.quantum_device = quantum_device

        # embeds vocab (centroids) into model dimension
        self.embed_tokens = nn.Embedding(self.vocab_size, self.embed_dim)
        # positional embedding
        self.pos_embed = nn.Embedding(self.max_seq_len, self.embed_dim)
        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(self.embed_dim))
        nn.init.normal_(self.sos)

        # transformer
        self.transformer_layer = HybridTransformer(self.embed_dim, self.n_heads, self.quantum_device)
        self.transformer = nn.ModuleList([self.transformer_layer for _ in range(self.n_layers)])

        # layer norm
        self.ln_f = nn.LayerNorm(self.embed_dim)
        # final linear layer to project to vocab size
        self.head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

    def forward(self, x):

        # x: (B, S) where B is batch size and S is sequence length
        batch_size, seq_len = x.size()
        device = x.device

        # embed each part of sequence with vector of size embed_dim
        h = self.embed_tokens(x)

        # embed tokens and postions
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (B, S)
        h =  h + self.pos_embed(positions)  # (B, S, embed_dim)

        # transformer layers
        # note that I do not need to specify mask becuase it is built into MultiHeadQMSAN
        for layer in self.transformer:
            h = layer(h)

        # final layer norm and head
        out = self.ln_f(h)
        logits = self.head(out)
        return logits