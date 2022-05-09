import datetime
import logging
import math
import os
import re
import time
import traceback
import shutil
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

from allennlp.common.util import int_to_device

import torch
import torch.distributed as dist
from torch.cuda import amp
import torch.optim.lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from allennlp.training.callbacks import TrainerCallback
from allennlp.data import DataLoader, TensorDict
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer

logger = logging.getLogger(__name__)


def progress_generator(max_value):
    current = 0
    while True:
        current += 1
        yield min(current, max_value) / max_value


@TrainerCallback.register("gradient_lambda_callback")
class GradientLambdaCallback(TrainerCallback):
    """
    A callback that you pass to the `GradientDescentTrainer` to access a variable lambda value
    for training with a gradient reversal signal. This sets a lambda attribute accessible
    within the model for the backwards pass of the gradient reversal layer.
    """
    def __init__(self,
                 serialization_dir: str,
                 num_epochs: int,
                 batches_per_epoch: int,
                 gamma: int = 10):
        super().__init__(serialization_dir)
        self._generator_max = num_epochs * batches_per_epoch
        self._progress_generator = progress_generator(self._generator_max)  # Does not handle resuming training
        self._gamma = gamma

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        super().on_start(trainer, is_primary)
        trainer.model._gradient_reverse_lambda = 0

    def _get_lambda_from_progress(self, progress: float = 0.):
        return (2 / (1 + math.exp(-self._gamma * progress))) - 1

    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        This callback hook is called after the end of each batch.
        """
        progress = next(self._progress_generator)
        lambda_val = self._get_lambda_from_progress(progress)
        trainer.model._gradient_reverse_lambda = lambda_val
