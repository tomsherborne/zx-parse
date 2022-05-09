from typing import Dict, List, Optional
from collections import defaultdict

import torch
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import Activation
from allennlp.data.fields import MetadataField, TensorField
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util, InitializerApplicator

from allennlp_models.generation.modules.seq_decoders.seq_decoder import SeqDecoder
from allennlp_models.generation.models.composed_seq2seq import ComposedSeq2Seq
from pretrain_code.modules import GradientReversalFunction

SQL_TASK = "sql"
NLG_TASK = "nlg"
DOM_TASK = "dom"


@Model.register("seq2seq_singlehead_lambda")
class Seq2SeqSingleHeadLambda(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            source_text_embedder: TextFieldEmbedder,
            encoder: Seq2SeqEncoder,
            decoder_sql: SeqDecoder,
            loss_weights: Dict[str, float] = None,
            use_gradient_reversal: bool = False,
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)

        self._source_text_embedder = source_text_embedder
        self._encoder = encoder
        self._tasks = ["sql", "nlg", "dom"]
        self._decoder_sql = decoder_sql
        self._loss_weights = loss_weights or defaultdict(lambda: 1.0)

        if self._encoder.get_output_dim() != self._decoder_sql.get_output_dim():
            raise ConfigurationError(
                f"Encoder output dimension {self._encoder.get_output_dim()} should be"
                f" equal to decoder dimension {self._decoder_sql.get_output_dim()}."
            )

        self._use_gradient_reversal = use_gradient_reversal
        if self._use_gradient_reversal:
            self._gradient_reverse_lambda = 0
            self._gradient_reverse_loss = torch.nn.CrossEntropyLoss(reduction="mean")
            self._domain_predictor_ff = torch.nn.Linear(in_features=self._encoder.get_output_dim(), out_features=6)
        initializer(self)

    @overrides
    def forward(
            self,  # type: ignore
            source_tokens: TextFieldTensors,
            target_tokens: TextFieldTensors = None,
            source_lang: TensorField = None,
            task: MetadataField = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Handle `task` MetadataField- A sequence of `str` representing each task
        # We assume HomogenousRoundRobin scheduling so we need to assert all equal
        loss = None
        assert [t == t[0] for t in task], f"All items in {task} are not equal."
        assert task[0] in self._tasks, f"{task[0]} is not a valid task ({self._tasks})"
        batch_task = task[0]

        # Encode source tokens
        state = self._encode(source_tokens)
        outputs = {**state}

        if batch_task == SQL_TASK:
            head_output = self._decoder_sql(state, target_tokens)
            for key in head_output:
                outputs[f"sql_{key}"] = head_output[key]
        else:
            head_output = {}

        if "loss" in head_output:
            loss = self._loss_weights[batch_task] * head_output['loss']

        if loss is not None:
            outputs['loss'] = loss

        if self.training and self._use_gradient_reversal:
            # Additional domain selection task
            domain_prediction_loss = self._domain_prediction(state, source_lang)

            if domain_prediction_loss is not None:
                if 'loss' in outputs:
                    outputs['loss'] += self._loss_weights[DOM_TASK] * domain_prediction_loss
                else:
                    outputs['loss'] = self._loss_weights[DOM_TASK] * domain_prediction_loss

                outputs['lp_loss'] = domain_prediction_loss

        return outputs

    @overrides
    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # For SQL decoder keys:
        sql_outputs = {}
        for key, value in output_dict.items():
            if key.startswith("sql"):
                sql_outputs[key.replace(f"sql_", "")] = value
        readable_sql_outputs = self._decoder_sql.post_process(sql_outputs)
        for key, value in readable_sql_outputs.items():
            output_dict[f"sql_{key}"] = value

        if "encoder_outputs" in output_dict:
            del output_dict['encoder_outputs']

        if "source_mask" in output_dict:
            del output_dict['source_mask']

        return output_dict

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Make forward pass on the encoder.

        # Parameters

        source_tokens : `TextFieldTensors`
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.

        # Returns

        Dict[str, torch.Tensor]
            Map consisting of the key `source_mask` with the mask over the
            `source_tokens` text field,
            and the key `encoder_outputs` with the output tensor from
            forward pass on the encoder.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_text_embedder(source_tokens)

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)

        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _domain_prediction(self, encoder_state: Dict[str, torch.Tensor], locale_labels: torch.Tensor):
        """
        Predict the domain from possible domains and return loss

        :param encoder_state:
        :return:
        """
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = encoder_state['encoder_outputs']

        # shape: (batch_size, max_input_sequence_length)
        source_mask = encoder_state['source_mask']

        # shape (batch_size, 1, encoder_output_dim)
        # Compress the encoding into a 1D sentence representation
        encoder_outputs = torch.mul(encoder_outputs, source_mask.float().unsqueeze(-1))
        encoder_outputs = torch.unsqueeze(encoder_outputs.sum(dim=-2) / source_mask.sum(dim=-1, keepdim=True), dim=-2)

        # Grad reversal layer
        encoder_outputs = GradientReversalFunction.apply(encoder_outputs, self._gradient_reverse_lambda)

        # Feedforward layer
        ff_output = self._domain_predictor_ff(encoder_outputs).squeeze()

        return self._gradient_reverse_loss(ff_output, locale_labels.squeeze())

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        sql_metrics = self._decoder_sql.get_metrics(reset)
        sql_metrics = {"sql_" + k: v for k, v in sql_metrics.items()}
        return sql_metrics

    default_predictor = "seq2seq_multihead"
