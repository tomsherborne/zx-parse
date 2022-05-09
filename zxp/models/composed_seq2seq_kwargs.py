from typing import Dict, Optional
import torch

from overrides import overrides
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator
from allennlp_models.generation.models import ComposedSeq2Seq
from allennlp_models.generation.modules.seq_decoders.seq_decoder import SeqDecoder

@Model.register("composed_seq2seq_kw")
class ComposedSeq2SeqKW(ComposedSeq2Seq):
    """
    Does everything that the standard composed_seq2seq model does but
    doesn't care if you pass additional arguments to forward()
    """
    def __init__(
        self,
        vocab: Vocabulary,
        source_text_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        decoder: SeqDecoder,
        tied_source_embedder_key: Optional[str] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super(ComposedSeq2SeqKW, self).__init__(vocab,
                                                source_text_embedder,
                                                encoder,
                                                decoder,
                                                tied_source_embedder_key,
                                                initializer,
                                                **kwargs)

    @overrides
    def forward(
        self,  # type: ignore
        source_tokens: TextFieldTensors,
        target_tokens: TextFieldTensors = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        return super().forward(source_tokens, target_tokens)
