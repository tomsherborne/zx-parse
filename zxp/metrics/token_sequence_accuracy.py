from typing import List, Dict

from overrides import overrides

import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics import Metric


@Metric.register("token_sequence_accuracy")
class TokenSequenceAccuracy(Metric):
    """
    Simple sequence accuracy based on tokens, as opposed to tensors.
    """

    def __init__(self) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.

    @overrides
    def reset(self) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.

    @overrides
    def __call__(self,
                 predictions: List[List[str]],
                 gold_targets: List[List[str]]) -> None:
        # This is actually a no-op
        predictions, gold_targets = self.detach_tensors(predictions, gold_targets)

        _total_count = len(predictions)
        _correct_count = 0
        for predicted_tokens, gold_tokens in zip(predictions, gold_targets):
            predicted_tokens = [p for p in predicted_tokens if p]
            gold_tokens = [g for g in gold_tokens if g]
            if predicted_tokens == gold_tokens:
                _correct_count += 1

        if is_distributed():
            device = torch.device("cuda" if dist.get_backend() == "nccl" else "cpu")
            correct_count = torch.tensor(_correct_count).to(device)
            total_count = torch.tensor(_total_count).to(device)
            dist.all_reduce(correct_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            _correct_count = correct_count.item()
            _total_count = total_count.item()

        self._correct_counts = _correct_count
        self._total_counts += _total_count

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if self._total_counts == 0:
            accuracy = 0.
        else:
            accuracy = self._correct_counts / self._total_counts

        if reset:
            self.reset()

        return {"seq_acc": accuracy}
