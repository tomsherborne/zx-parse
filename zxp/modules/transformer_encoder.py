import math
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides
from torch import nn
from torch.autograd import Variable

from allennlp.modules.layer_norm import LayerNorm
from allennlp.nn import util as nn_util
from allennlp.nn import Activation
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp_models.lm.modules.seq2seq_encoders.bidirectional_lm_transformer import (
    SublayerConnection,
    subsequent_mask,
    PositionwiseFeedForward,
    PositionalEncoding,
    MultiHeadedAttention,
)

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiHeadedAttention):
        for linear in module.linears:
            linear.weight.data.normal_(mean=0.0, std=0.02)


class PositionwiseFeedForwardMultiActivation(torch.nn.Module):
    """Implements FFN equation."""

    def __init__(self, input_dim: int, ff_dim: int, activation: Activation, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1 = torch.nn.Linear(input_dim, ff_dim)
        self.w_2 = torch.nn.Linear(ff_dim, input_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))



@Seq2SeqEncoder.register("transformer_encoder")
class LanguageModelTransformer(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        feedforward_hidden_dim: int,
        num_layers: int,
        num_attention_heads: int,
        use_positional_encoding: bool = True,
        positional_encoding_max_steps: int = 1024,
        dropout_prob: float = 0.1,
        residual_dropout_prob: float = 0.2,
        attention_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()

        attn = MultiHeadedAttention(num_attention_heads, input_dim, attention_dropout_prob)
        feed_forward = PositionwiseFeedForwardMultiActivation(input_dim=input_dim,
                                                              ff_dim=feedforward_hidden_dim,
                                                              activation=Activation.by_name("gelu")(),
                                                              dropout=dropout_prob)
        self._input_dim = input_dim
        self._output_dim = input_dim

        self._embed_scale = math.sqrt(input_dim)
        self._positional_embedder = (
            PositionalEncoding(input_dim, positional_encoding_max_steps)
            if use_positional_encoding
            else None
        )
        self._dropout = nn.Dropout(dropout_prob)
        self._self_attention = TransformerEncoder(
            EncoderLayer(
                input_dim, deepcopy(attn), feed_forward, residual_dropout_prob
            ),
            num_layers,
        )

        self.apply(init_bert_params)

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self, token_embeddings: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:

        # Positional embeddings
        if self._positional_embedder:
            token_embeddings = self._positional_embedder(token_embeddings)

        # Dropout token embeddings
        token_embeddings = self._dropout(token_embeddings)

        # Shape (batch_size, timesteps, timesteps)
        mask = mask.unsqueeze(-2)

        # Encoder layers
        encoder_output = self._self_attention(token_embeddings, mask)

        return encoder_output


class TransformerEncoder(torch.nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(
        self, layer: torch.nn.Module, num_layers: int, return_all_layers: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn_util.clone(layer, num_layers)
        self.norm = LayerNorm(layer.size)
        self.return_all_layers = return_all_layers

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        all_layers = []
        for layer in self.layers:
            x = layer(x, mask)
            if self.return_all_layers:
                all_layers.append(x)

        if self.return_all_layers:
            all_layers[-1] = self.norm(all_layers[-1])
            return all_layers
        return self.norm(x)


class EncoderLayer(torch.nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn_util.clone(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self,
                x: torch.Tensor,
                mask: torch.BoolTensor) -> torch.Tensor:
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask))
        return self.sublayer[1](x, self.feed_forward)

