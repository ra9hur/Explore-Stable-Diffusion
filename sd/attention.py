import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        # in_proj_bias, out_proj_bias
        # biases for the W matrices not present in the original Transformer

        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        # let's define the W matrices - wq, wk and wv 
        # This is represented as one big linear layer instead of representing it as 3 diff matrices
        # we just say that it's a big  Matrix - 3 * d_embed
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias = in_proj_bias)

        # This one represents Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads    # dimension of head


    def forward(self, x, causal_mask=False):
        # x: (batch_size, seg_len, dim)        (batch_size, height * width, features)

        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim * 3) -> 
        # 3 tensors of shape (batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, H, dim / H) ->
        # (batch_size, H, seq_len, dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (batch_size, H, seq_len, dim / H) @ (batch_size, H, dim / H, seq_len) ->
        # (batch_size, H, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)        # matrix multiply

        # mask is something that we apply when we calculate the attention 
        # if we don't want two tokens to relate to each other, 
        # we basically substitute interaction with minus infinity in matrix before applying soft max
        if causal_mask:
            # mask where upper triangle (above principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf)

        # Divide by d_k (dim / H)
        weight /= math.sqrt(self.d_head)

        # Applying softmax
        weight = F.softmax(weight, dim=-1)

        # (batch_size, H, seq_len, seq_len) @ (batch_size, H, seq_len, dim / H) ->
        # (batch_size, H, seq_len, dim / H)
        output = weight @ v                     # matrix multiply

        # (batch_size, H, seq_len, dim / H) -> (batch_size, seq_len, H, dim / H)
        output = output.transpose(1, 2)

        # (batch_size, seq_len, H, dim / H) -> (batch_size, seq_len, dim)
        output = output.reshape(input_shape)    # multiply by Wo weight matrix

        output = self.out_proj(output)

        return output
    


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embd, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embd, d_embd, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embd, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embd, bias=in_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embd // n_heads

    def forward(self, x, y):
        # x latent: (batch_size, seq_len_q, dim_q)
        # y context: (batch_size, seq_len_kv, dim_kv) = (batch_size, 77, 768)

        input_shape = x.shape
        batch_size, seq_len, d_embd = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # (batch_size, seq_len_q, dim_q) -> (batch_size, seq_len_q, h, dim_q / h) -> (batch_size, h, seq_len_q, dim_q / h)
        q = q.view(interim_shape).transpose(1, 2)
        # (batch_size, seq_len_kv, dim_kv) -> (batch_size, seq_len_kv, h, dim_kv / h) -> (batch_size, h, seq_len_kv, dim_kv / h)
        k = k.view(interim_shape).transpose(1, 2)
        # (batch_size, seq_len_kv, dim_kv) -> (batch_size, seq_len_kv, h, dim_kv / h) -> (batch_size, h, seq_len_kv, dim_kv / h)
        v = v.view(interim_shape).transpose(1, 2)

        # (batch_size, h, seq_len_q, dim_q / h) @ (batch_size, h, dim_kv / h, seq_len_kv) -> (batch_size, h, seq_len_q, seq_len_kv)
        weight = q @ k.transpose(-1, -2)

        # (batch_size, h, seq_len_q, seq_len_kv)
        weight /= math.sqrt(self.d_head)

        # (batch_size, h, seq_len_q, seq_len_kv)
        weight = F.softmax(weight, dim=-1)

        # (batch_size, h, seq_len_q, seq_len_kv) @ (batch_size, h, seq_len_kv, dim_kv / h) -> (batch_size, h, seq_len_q, dim_q / h)
        output = weight @ v

        # (batch_size, h, seq_len_q, dim_q / h) -> (batch_size, seq_len_q, h, dim_q / h)
        output = output.transpose(1, 2).contiguous()

        # (batch_size, seq_len_q, h, dim_q / h) -> (batch_size, seq_len_q, dim_q)
        output = output.view(input_shape)

        # (batch_size, seq_len_q, dim_q)
        output = self.out_proj(output)

        # (batch_size, seq_len_q, dim_q)
        return output
    
    