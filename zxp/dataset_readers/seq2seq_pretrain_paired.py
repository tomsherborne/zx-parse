import csv
from typing import Dict, List, Optional
import logging
import copy
from random import randint, sample

from overrides import overrides
import torch
from transformers import MBart50TokenizerFast

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import Token
from allennlp.data.fields import TextField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, PretrainedTransformerIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

VALID_MBART50_MODELS = ["facebook/mbart-large-50-many-to-many-mmt"]
VALID_MBART_MODELS = ["facebook/mbart-large-cc25"]
MASK_SYMBOL = "<mask>"
LANG2IDX = {"en_XX": torch.LongTensor([0]), "de_DE": torch.LongTensor([1]), "zh_CN": torch.LongTensor([2]),
            "fr_XX": torch.LongTensor([3]), "es_XX": torch.LongTensor([4]), "pt_XX": torch.LongTensor([5])}
DEFAULT_LANGIDX = torch.LongTensor([0])


@DatasetReader.register("seq2seq_pretrain_paired")
class PretrainedTransformerSeq2SeqPairedDatasetReader(DatasetReader):
    """
    Unifies the seq2seq dataset parsers for standard Huggingface embedders (BART, XLM-Roberta, etc)
    with the embedder for mBART-50 which requires a different interface due to locale switching.
    Assume NL-SQL-LOCALE triples and SQL tokenization differs.

    Compose Instances of "source_tokens", "source_lang" and optionally "target_tokens".
        - "source_tokens" should be NL
        - "source_lang" should be an ISO code (or converted to one)
        - "target_tokens" will be SQL.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>\t<source_lang>
    If we lack <source_lang> then we assume this is "en_XX".

    The output of `read` is a list of `Instance` s with the fields:
        source_tokens : `TextField` and
        target_tokens : `Optional[TextField]` and
        source_lang   : `TextField`
    """

    def __init__(
            self,
            source_pretrained_model_name: str = None,
            source_token_namespace: str = "tokens",
            use_mbart_indexer: bool = True,
            target_tokenizer: Tokenizer = None,
            target_token_indexers: Dict[str, TokenIndexer] = None,
            target_add_start_token: bool = True,
            target_add_end_token: bool = True,
            target_start_symbol: str = START_SYMBOL,
            target_end_symbol: str = END_SYMBOL,
            delimiter: str = "\t",
            source_max_tokens: Optional[int] = 1024,
            target_max_tokens: Optional[int] = None,
            quoting: int = csv.QUOTE_MINIMAL,
            num_tokens_to_mask: int = 0,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._mbart50_tokenizer = False
        self._mbart_tokenizer = False
        if source_pretrained_model_name in VALID_MBART50_MODELS:
            logger.info(f"Creating mBART-50 based tokenizer for {source_pretrained_model_name}")
            self._source_tokenizer = MBARTTokenizerWrapper(source_pretrained_model_name, source_max_tokens)
            self._mbart50_tokenizer = True
        else:
            logger.info(f"Creating generic HuggingFace tokenizer for {source_pretrained_model_name}")
            # Tokenization works best if we use the ANLP implementation. For mbart-large-cc25 we need
            # to manually switch the source lang before each tokenizer call.
            if source_pretrained_model_name in VALID_MBART_MODELS:
                self._mbart_tokenizer = True

            self._source_tokenizer = PretrainedTransformerTokenizer(
                model_name=source_pretrained_model_name, add_special_tokens=True)

        self._source_token_indexers = {
            source_token_namespace: PretrainedTransformerIndexer(model_name=source_pretrained_model_name,
                                                                 namespace=source_token_namespace
                                                                 )
        }

        # Language code validator
        self._validator = LanguageFormatter(self._source_tokenizer.tokenizer.additional_special_tokens)

        if use_mbart_indexer:
            self._source_token_indexers = {
                source_token_namespace: PretrainedTransformerIndexer(model_name=source_pretrained_model_name,
                                                                     namespace=source_token_namespace
                                                                     )
            }
        else:
            self._source_token_indexers = {
                source_token_namespace: SingleIdTokenIndexer(namespace=source_token_namespace)
            }

        assert type(target_tokenizer) is not None

        if type(target_tokenizer) is PretrainedTransformerTokenizer:
            # TODO(tom): Practically this event shouldn't happen.
            #  Why would tgt be pretrained differently to src?
            raise ValueError("Debug error: Tokenization has gone wrong and needs fixing...")

        logger.info(f"Target tokenizer generically declared as {type(target_tokenizer).__name__}")
        self._target_tokenizer = target_tokenizer

        # Get BOS and EOS symbols
        bos_eos_symbols = self._target_tokenizer.tokenize(
            target_start_symbol + " " + target_end_symbol
        )
        if len(bos_eos_symbols) != 2:
            raise ValueError(f"Target sequence tokens not correctly fetched. Output was {bos_eos_symbols}")
        self._start_token, self._end_token = bos_eos_symbols

        # Start and end token logic (probably always True)
        self._target_add_start_token = target_add_start_token
        self._target_add_end_token = target_add_end_token

        # Locate mask token
        self._num_tokens_to_mask = num_tokens_to_mask
        mask_seq = self._source_tokenizer.tokenize(MASK_SYMBOL)
        mask_seq_ = [t for t in mask_seq if t.text == MASK_SYMBOL]
        if mask_seq_:
            self._mask_token = mask_seq_[0]
        else:
            raise ValueError(f"Cannot locate mask token inside source tokenizer. Search over {mask_seq}.")

        logger.info(f"Target tokenizer BOS: \"{self._start_token}\" and EOS: \"{self._end_token}\"")

        # Target indexing should probably not match source as we aren't copying the embedder.
        self._target_token_indexers = target_token_indexers

        # TSV delimiter
        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.quoting = quoting

    @overrides
    def _read(self, file_path: str):

        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0

        # Open data file
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Enumerate rows in data file
            for line_num, row in enumerate(
                    csv.reader(data_file, delimiter=self._delimiter, quoting=self.quoting)
            ):
                # Expected format NL\tLF\tLOCALE
                if len(row) == 3:
                    source_sequence, target_sequence, source_lang = row
                else:
                    raise ConfigurationError(
                        "Invalid line format for paired data with locale: %s (line number %d)" % (row, line_num + 1)
                    )

                yield self.text_to_instance(source_sequence, target_sequence, source_lang)

        if self._source_max_tokens and self._source_max_exceeded:
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                self._source_max_exceeded,
                self._source_max_tokens,
            )
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info(
                "In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                self._target_max_exceeded,
                self._target_max_tokens,
            )

    @overrides
    def text_to_instance(
            self, source_string: str, target_string: str = None, source_lang: str = None
    ) -> Instance:  # type: ignore
        source_lang = self._validator(source_lang)
        tokenizer_args = (source_string, source_lang) if self._mbart50_tokenizer else (source_string, )
        tokenized_source = self._source_tokenizer.tokenize(*tokenizer_args)

        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_exceeded += 1
            tokenized_source = tokenized_source[: self._source_max_tokens]

        # Replace between `0` and `num_symbols_to_mask` symbols with the <mask> token
        if self._num_tokens_to_mask:
            # We check that we aren't masking the full sequence. If we are trying to mask more tokens
            # than available then we set a maximum of half the sequence
            max_num_tokens_to_mask = self._num_tokens_to_mask if \
                len(tokenized_source) > self._num_tokens_to_mask else int(len(tokenized_source) / 2)

            num_mask = randint(0, max_num_tokens_to_mask) # Replace between 0 and _num_token_to_mask tokens
            idx_to_mask = sample(range(len(tokenized_source)), num_mask)
            for idx in idx_to_mask:
                tokenized_source[idx] = self._mask_token

        source_field = TextField(tokenized_source, self._source_token_indexers)
        source_lang_iso = self._validator(source_lang)
        lang_field = TensorField(LANG2IDX.get(source_lang_iso, DEFAULT_LANGIDX))
        # Passing in target_string as a copy of the source is redundant but keeps the inference logic nicer
        # as only testing will skip this statement. If self._no_target_sequence then we don't use it.
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)

            if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                self._target_max_exceeded += 1
                tokenized_target = tokenized_target[: self._target_max_tokens]

            if self._target_add_start_token:
                tokenized_target.insert(0, copy.deepcopy(self._start_token))
            if self._target_add_end_token:
                tokenized_target.append(copy.deepcopy(self._end_token))

            target_field = TextField(tokenized_target, self._target_token_indexers)

            return Instance({"source_tokens": source_field, "target_tokens": target_field, "source_lang": lang_field})
        else:
            return Instance({"source_tokens": source_field, "source_lang": lang_field})


class MBARTTokenizerWrapper(object):
    def __init__(self, pretrained_model_name, max_tokens):
        self.tokenizer = MBart50TokenizerFast.from_pretrained(pretrained_model_name)
        self._max_tokens = max_tokens

    def tokenize(self, source_string: str, source_lang: str = "en_XX", add_special_tokens=True) -> List[Token]:
        """
        This method only handles a single sentence (or sequence) of text.
        Modified from Allennlp.data.tokenizer.pretrainedtransformertokenizer.
        1) Assumes the fast implementation of MBart50 so omits need for self._estimate_character_indices()
        2) Sets the src_lang for the tokenizer before the function
        """
        max_length = self._max_tokens

        self.tokenizer.src_lang = source_lang

        encoded_tokens = self.tokenizer(
            text=source_string,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=True,
            stride=0,
            return_tensors=None,
            return_offsets_mapping=self.tokenizer.is_fast,
            return_attention_mask=False,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )

        # token_ids contains a final list with ids for both regular and special tokens
        token_ids, token_type_ids, special_tokens_mask, token_offsets = (
            encoded_tokens["input_ids"],
            encoded_tokens["token_type_ids"],
            encoded_tokens["special_tokens_mask"],
            encoded_tokens["offset_mapping"],
        )

        tokens = []
        for token_id, token_type_id, special_token_mask, offsets in zip(
                token_ids, token_type_ids, special_tokens_mask, token_offsets
        ):
            if offsets is None or offsets[0] >= offsets[1]:
                start = None
                end = None
            else:
                start, end = offsets

            tokens.append(
                Token(
                    text=self.tokenizer.convert_ids_to_tokens(token_id, skip_special_tokens=False),
                    text_id=token_id,
                    type_id=token_type_id,
                    idx=start,
                    idx_end=end,
                )
            )
        return tokens


class LanguageFormatter:
    """
    Function to format a generic language code to a valid language ID string
    accepted by an mBART tokenizer.

    drop_pt is a substitution to change pt_XX to es_XX as mBART-large-cc25
    does not include portuguese.
    """
    def __init__(self, valid_langs: List):
        super(LanguageFormatter, self).__init__()
        self._valid_langs = valid_langs
        self._lang2iso_map = {"en": "en_XX", "zh_cn": "zh_CN", "zh": "zh_CN", "pt": "pt_XX",
                              "fr": "fr_XX", "de": "de_DE", "es": "es_XX"}
        self._drop_pt = False
        if "pt_XX" not in self._valid_langs:
            self._drop_pt = True

    def __call__(self, lang_code: str) -> str:
        # Set source lang prior to any encoding
        source_lang = self._lang2iso_map.get(lang_code, lang_code)  # Transform the source lang else remain unchanged
        if source_lang == "pt_XX" and self._drop_pt:
            source_lang = "es_XX"
        if self._valid_langs and source_lang not in self._valid_langs:
            raise ValueError(f"Source lang {source_lang} not recognised. "
                             f"Valid source languages are {self._valid_langs}")
        return source_lang
