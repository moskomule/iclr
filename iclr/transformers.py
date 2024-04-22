"""
Based on https://github.com/karpathy/nanoGPT/blob/master/model.py

Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from typing import Callable

import torch
import torch.nn as nn
from rich import print
from torch.nn import functional as F


def attn_scale_len(k: torch.Tensor) -> float:
    # tf-as-st style
    return 1.0 / k.size(-2)


def attn_scale_hs_sqrt(k: torch.Tensor) -> float:
    # standard style
    return 1.0 / math.sqrt(k.size(-1))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


def self_attention(
        k: torch.Tensor,
        q: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        enable_causal_mask: bool,
        activation: Callable[[torch.Tensor], torch.Tensor],
        attn_scale: Callable[[torch.Tensor], float]
) -> tuple[torch.Tensor, torch.Tensor]:
    T = k.size(-2)
    att = (q @ k.transpose(-2, -1)) * attn_scale(k)
    if enable_causal_mask:
        att = att.masked_fill(bias[:, :, :T, :T] == 0, float('-inf'))
    att = activation(att) @ v
    y = att @ v
    return y, att


class SelfAttention(nn.Module):

    def __init__(self,
                 dim_embed: int,
                 num_heads: int,
                 activation: Callable[[torch.Tensor], torch.Tensor],
                 attn_scale: Callable[[torch.Tensor], float],
                 flash_attention: bool,
                 enable_bias: bool,
                 enable_causal_mask: bool,
                 enable_proj: bool,
                 num_tokens: int,
                 ):
        super().__init__()
        assert dim_embed % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(dim_embed, 3 * dim_embed, bias=enable_bias)
        # output projection
        self.c_proj = nn.Linear(dim_embed, dim_embed, bias=enable_bias) if enable_proj else nn.Identity()

        self.num_heads = num_heads
        self.num_embed = dim_embed
        self.self_attention = self_attention
        self.activation = activation
        self.attn_scale = attn_scale
        self.flash_attention = flash_attention

        self.enable_causal_mask = enable_causal_mask
        self.bias = None
        if self.enable_causal_mask:
            self.register_buffer("bias", torch.tril(torch.ones(num_tokens, num_tokens))[None, None])

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.num_embed, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        if self.flash_attention:
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=0,
                                                                 is_causal=self.enable_causal_mask)
        else:
            # to get attention matrix
            y, att = self.self_attention(k, q, v, self.bias, self.enable_causal_mask, self.activation, self.attn_scale)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self,
                 dim_embed: int,
                 activation: Callable[[torch.Tensor], torch.Tensor],
                 enable_bias: bool,
                 ):
        super().__init__()
        self.c_fc = nn.Linear(dim_embed, 4 * dim_embed, bias=enable_bias)
        self.c_proj = nn.Linear(4 * dim_embed, dim_embed, bias=enable_bias)
        self.activation = activation

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim_embed: int,
                 num_heads: int,
                 attn_activation: Callable[[torch.Tensor], torch.Tensor],
                 mlp_activation: Callable[[torch.Tensor], torch.Tensor],
                 attn_scale: Callable[[torch.Tensor], float],
                 num_tokens: int,
                 enable_ln: bool,
                 enable_bias: bool,
                 enable_proj: bool,
                 enable_causal_mask: bool,
                 enable_flash_attention: bool):
        super().__init__()

        self.ln_1 = LayerNorm(dim_embed, bias=enable_bias) if enable_ln else nn.Identity()
        self.ln_2 = LayerNorm(dim_embed, bias=enable_bias) if enable_ln else nn.Identity()
        self.attn = SelfAttention(dim_embed, num_heads, attn_activation, attn_scale, enable_flash_attention,
                                  enable_bias, enable_causal_mask, enable_proj, num_tokens)
        self.mlp = MLP(dim_embed, mlp_activation, enable_bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class STTransformer(nn.Module):
    """ Transformer model in TFs as Statisticians
    """

    def __init__(self,
                 num_layers: int,
                 dim_embed: int,
                 num_heads: int,
                 attn_activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
                 mlp_activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
                 attn_scale: Callable[[torch.Tensor], float] = attn_scale_len,
                 num_tokens: int = None,
                 enable_ln: bool = False,
                 enable_bias: bool = False,
                 enable_proj: bool = False,
                 enable_causal_mask: bool = False,
                 enable_flash_attention: bool = False
                 ):
        super().__init__()

        assert (not enable_causal_mask) or (num_tokens is not None), "Cannot enable causal mask when num_tokens is None"

        self.dim_embed = dim_embed
        self.blocks = nn.ModuleList([Block(dim_embed, num_heads, attn_activation, mlp_activation, attn_scale,
                                           num_tokens, enable_ln, enable_bias, enable_proj, enable_causal_mask,
                                           enable_flash_attention) for _ in range(num_layers)])
        self.ln = LayerNorm(dim_embed, enable_bias) if enable_ln else nn.Identity()
        self.readout = nn.Linear(dim_embed, 1, bias=False)

        # init all weights
        self._init_weights()

        # report number of parameters
        print(f"number of parameters: {self.num_params:.3e}")

    @property
    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self):
        def f(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.apply(f)

    def forward(self,
                xs: torch.Tensor,
                ys: torch.Tensor
                ) -> torch.Tensor:
        # xs: input (b, t, c)
        # ys: target (b, t)
        ys = ys.clone()
        ys[:, -1] = 0  # because test data!
        h = torch.cat([xs, ys[:, :, None]], dim=-1)
        for block in self.blocks:
            h = block(h)
        h = self.ln(h)
        return self.readout(h)

    def configure_optimizers(self,
                             weight_decay: float,
                             learning_rate: float,
                             betas: tuple[float, float] = None,
                             ):
        # start with all the candidate parameters
        param_dict = [p for pn, p in self.named_parameters() if p.requires_grad]
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in param_dict if p.dim() >= 2]
        nodecay_params = [p for p in param_dict if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=torch.cuda.is_available())
        return optimizer
