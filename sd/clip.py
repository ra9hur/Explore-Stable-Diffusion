import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        
        # position encoding in org transformer are given by sin functions
        # here learned parameters are used
        # learned during training that provide position of the token
        # A learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))


    def forward(self, tokens):
        # batch_size, seq_len) -> (batch_size, seq_len, dim)
        x = self.token_embedding(tokens)
        # batch_size, seq_len) -> (batch_size, seq_len, dim)
        x += self.position_embedding

        return x



class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        # Pre-attention norm
        self.layernorm_1 = nn.LayerNorm(n_embd)

        # Self Attention
        self.attention = SelfAttention(n_head, n_embd)

        # Pre-FNN norm
        self.layernorm_2 = nn.LayerNorm(n_embd)

        # Feedforward layer
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)


    def forward(self, x):
        # (batch_size, seq_len, dim)
        residue = x

        ### SELF ATTENTION ###
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        ### Feedforward Layer ###
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension

        residue = x

        x = self.layernorm_2(x)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, 4 * dim)
        x = self.linear_1(x)

        # Activation function - QuickGELU
        # But why ?
        x = x * torch.sigmoid(1.702 * x)

        # (batch_size, seq_len, 4 * dim) -> (batch_size, seq_len, dim)
        x = self.linear_2(x)

        x += residue

        return x



class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # Vocabulary size is 49408, referring the download file
        # Embedding size in the org transformer is 512, here it is 768
        # the maximum sequence length considering padding as well is 77
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            # 12 indicates # of heads in multi-headed attention
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)


    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # long tensor since the input IDs are usually numbers that indicate the position of each token inside of the vocabulary
        tokens = tokens.type(torch.long)

        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        # seq_len - 77, dim - 768
        state = self.embedding(tokens)

        # Apply encoder layers similar to the Transformer's encoder
        for layer in self.layers:
            state = layer(state)
        
        output = self.layernorm(state)

        # (batch_size, seq_len, dim)
        return output
    