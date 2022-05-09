import collections
from typing import Type, List, Dict
import logging

from overrides import overrides

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.models.model import Model
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import sanitize
from allennlp.data.fields import MetadataField
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers import DatasetReader, MultiTaskDatasetReader

logger = logging.getLogger(__name__)


@Predictor.register("seq2seq_anyhead")
class Seq2SeqAnyHeadPredictor(Predictor):
    """
    An intentionally fiddly Predictor Class that can handle MultiTask or non-Multitask Dataset readers
    during prediction. This avoids having different declaration scripts for inference as we call this
    predictor and the internal logic works out if we need to behave as a MultiTask reader or not.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._is_multihead = True if isinstance(dataset_reader, MultiTaskDatasetReader) else False

        if self._is_multihead:
            logger.info("MultiTaskDatasetReader detected. Predictor is set up as multi-head")
            self._json_to_instance = self._json_to_instance_multihead
            self.predict_instance = self._predict_instance_multihead
        else:
            logger.info(f"{type(dataset_reader)} detected. Predictor is set up as single-head")
            self._json_to_instance = self._json_to_instance_single

    def _predict_instance_multihead(self, instance: Instance) -> JsonDict:
        task_field = instance["task"]
        if not isinstance(task_field, MetadataField):
            raise ValueError(self._WRONG_FIELD_ERROR)
        task: str = task_field.metadata
        if not isinstance(self._dataset_reader, MultiTaskDatasetReader):
            raise ConfigurationError(self._WRONG_READER_ERROR)
        self._dataset_reader.readers[task].apply_token_indexers(instance)
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def _json_to_instance_multihead(self, json_dict: JsonDict) -> Instance:
        task = "sql"
        tti_args = (json_dict["source"], None, json_dict['source_lang'])
        instance = self._dataset_reader.readers[task].text_to_instance(*tti_args)
        return instance

    def _json_to_instance_single(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "...", 'vec': "..."}`.
        """
        tti_args = (json_dict["source"], None, json_dict['source_lang'])
        return self._dataset_reader.text_to_instance(*tti_args)

