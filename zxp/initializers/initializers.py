import logging
import re
import math
from typing import Callable, List, Tuple, Dict
import itertools
from overrides import overrides
import tarfile

import torch
import torch.nn.init

from allennlp.common import FromParams, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.nn.initializers import Initializer, PretrainedModelInitializer

logger = logging.getLogger(__name__)


@Initializer.register("pretrained_with_replacement")
class PretrainedModelReplacementInitializer(PretrainedModelInitializer):
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    This is an augmentation of PretrainedModelInitializer which uses string replacement
    to bulk rename components of old files to new files.

    e.g. for the argument {"parameter_name_overrides": {"encoder": "encoder_projection"}}
         all the parameters under old name encoder_projection e.g. "encoder_projection.sublayer[0].linear.weight"
         will be mapped to "encoder.sublayer[0].linear.weight"

    Use with extreme care.
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    Registered as an `Initializer` with name "pretrained_with_replacement".

    # Parameters

    weights_file_path : `str`, required
        The path to the weights file which has the pretrained model parameters.
    parameter_name_overrides : `Dict[str, str]`, optional (default = `None`)
        The mapping from the new parameter name to the name which should be used
        to index into the pretrained model parameters. If a parameter name is not
        specified, the initializer will use the parameter's default name as the key.
    """

    def __init__(
        self, weights_file_path: str, parameter_name_overrides: Dict[str, str] = None
    ) -> None:
        super(PretrainedModelReplacementInitializer, self).__init__(weights_file_path, parameter_name_overrides)

    @overrides
    def __call__(self, tensor: torch.Tensor, parameter_name: str, **kwargs) -> None:  # type: ignore
        for new_name_str in self.parameter_name_overrides:
            if new_name_str in parameter_name:
                old_name_str = self.parameter_name_overrides[new_name_str]
                logger.info(
                    f"Tensor {parameter_name} is loading weight with name replacement {new_name_str}:{old_name_str}"
                )
                parameter_name = parameter_name.replace(new_name_str, old_name_str)

        # If the size of the source and destination tensors are not the
        # same, then we need to raise an error
        source_weights = self.weights[parameter_name]

        if tensor.data.size() != source_weights.size():
            raise ConfigurationError(
                "Incompatible sizes found for parameter %s. "
                "Found %s and %s" % (parameter_name, tensor.data.size(), source_weights.size())
            )

        # Copy the parameters from the source to the destination
        tensor.data.copy_(source_weights.data)

