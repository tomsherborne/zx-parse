"""HuggingFace + PyTorch BART code ported to work as facets of AllenNLP models"""
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch.nn import CrossEntropyLoss
from allennlp.modules.layer_norm import LayerNorm

ACT2FN = {"relu": F.relu, "gelu": F.gelu}


class EncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 encoder_attention_heads,
                 encoder_ffn_dim,
                 encoder_final_ffn_output_dim=None,
                 activation_function="gelu",
                 normalize_before=False,
                 dropout=0.1,
                 activation_dropout=0.2,
                 attention_dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = Attention(self.embed_dim, encoder_attention_heads, dropout=attention_dropout)
        self.normalize_before = normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = dropout
        self.activation_fn = ACT2FN[activation_function]
        self.activation_dropout = activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_dim)

        if not encoder_final_ffn_output_dim:
            final_layer_dim = embed_dim
            self._drop_final_residual = False
        else:
            final_layer_dim = encoder_final_ffn_output_dim
            self._drop_final_residual = True
        self.fc2 = nn.Linear(encoder_ffn_dim, final_layer_dim)
        self.final_layer_norm = LayerNorm(final_layer_dim)

    def get_output_dim(self):
        return self.fc2.out_features

    def forward(self, x, encoder_padding_mask, output_attentions=False):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(
            query=x, key=x, key_padding_mask=encoder_padding_mask, output_attentions=output_attentions
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if not self._drop_final_residual:
            x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if torch.isinf(x).any() or torch.isnan(x).any():
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        return x, attn_weights


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Tensor]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state and static_kv:
                # previous time steps are cached - no need to recompute key and value if they are static
                key = None
        else:
            # this branch is hit by encoder
            saved_state = None

        q = self.q_proj(query) * self.scaling
        if static_kv and key is None:  # cross-attention with cache
            k = v = None
        elif static_kv and key is not None:  # cross-attention no prev_key found in cache
            k = self.k_proj(key)
            v = self.v_proj(key)
        else:  # self-attention
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state:
            k, v = self._concat_saved_state(k, v, saved_state, static_kv, bsz)

        # Update cache
        if isinstance(layer_state, dict):
            cached_shape = (bsz, self.num_heads, -1, self.head_dim)  # bsz must be first for reorder_cache
            layer_state[self.cache_key] = dict(prev_key=k.view(*cached_shape), prev_value=v.view(*cached_shape))

        src_len = k.size(1)
        assert key_padding_mask is None or key_padding_mask.shape == (bsz, src_len)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # Note: deleted workaround to get around fork/join parallelism not supporting Optional types. on 2020/10/15

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # make sure that attn_weights are included in graph
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

    def _concat_saved_state(self, k, v, saved_state, static_kv, bsz) -> Tuple[Tensor]:
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        prev_K = saved_state["prev_key"].view(bsz * self.num_heads, -1, self.head_dim)
        prev_V = saved_state["prev_value"].view(bsz * self.num_heads, -1, self.head_dim)
        new_K = prev_K if static_kv else torch.cat([prev_K, k], dim=1)
        new_V = prev_V if static_kv else torch.cat([prev_V, v], dim=1)
        return new_K, new_V