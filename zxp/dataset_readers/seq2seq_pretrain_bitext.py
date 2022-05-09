import csv
from typing import Dict, List, Optional
import logging
import copy
from random import randint, sample, random
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import Token
from allennlp.data.fields import TextField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, PretrainedTransformerIndexer, SingleIdTokenIndexer

from zxp.dataset_readers.seq2seq_pretrain_paired \
    import MBARTTokenizerWrapper, LANG2IDX, DEFAULT_LANGIDX, VALID_MBART50_MODELS, VALID_MBART_MODELS, LanguageFormatter

logger = logging.getLogger(__name__)
MASK_SYMBOL = "<mask>"


@DatasetReader.register("seq2seq_pretrain_bitext")
class PretrainedTransformerSeq2SeqBitextDatasetReader(DatasetReader):
    """
    Unifies the seq2seq dataset parsers for standard Huggingface embedders (BART, XLM-Roberta, etc)
    with the embedder for mBART-50 which requires a different interface due to locale switching.
    Here we have paired data and we randomly switch between L->L pairs and L->EN according to MT sample factor.

    Compose Instances of "source_tokens", "source_lang" and optionally "target_tokens".
        - "source_tokens" should be NL
        - "source_lang" should be an ISO code (or converted to one)
        - "target_tokens" will be NL.

    Expected format for each input line: <source_sequence_string>\t<source_lang>
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
            target_token_indexers: Dict[str, TokenIndexer] = None,
            delimiter: str = "\t",
            source_max_tokens: Optional[int] = 1024,
            quoting: int = csv.QUOTE_MINIMAL,
            num_tokens_to_mask: int = 0,
            maybe_translate_sample_factor: float = 0,
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

        # Language code validator
        self._validator = LanguageFormatter(self._source_tokenizer.tokenizer.additional_special_tokens)

        # There is a conflict here because we want to follow the source tokenization but DecoderNet instances
        # rigidly expect the START_SYMBOL, END_SYMBOL bookends. We try the HF approach with `add_special_tokens=False`
        if source_pretrained_model_name in VALID_MBART50_MODELS:
            logger.info(f"Creating mBART-50 based tokenizer for {source_pretrained_model_name}")
            self._target_tokenizer = MBARTTokenizerWrapper(source_pretrained_model_name, source_max_tokens)
        else:
            logger.info(f"Creating generic HuggingFace tokenizer for {source_pretrained_model_name}")
            self._target_tokenizer = PretrainedTransformerTokenizer(
                model_name=source_pretrained_model_name, add_special_tokens=False)

        logger.info(f"Expectation of sentence locale pairs without target sequence. tgt_tokenizer follows source.")

        # Target indexing should probably not match source as we aren't copying the embedder.
        self._target_token_indexers = target_token_indexers
        # DecoderNet instances expect these specific symbols.
        self._start_token = Token(START_SYMBOL)
        self._end_token = Token(END_SYMBOL)

        # Locate mask token
        self._num_tokens_to_mask = num_tokens_to_mask
        mask_seq = self._source_tokenizer.tokenize(MASK_SYMBOL)
        mask_seq_ = [t for t in mask_seq if t.text == MASK_SYMBOL]
        if mask_seq_:
            self._mask_token = mask_seq_[0]
        else:
            raise ValueError(f"Cannot locate mask token inside source tokenizer. Search over {mask_seq}.")

        # Start and end token logic
        self._target_add_start_token = True
        self._target_add_end_token = True

        logger.info(f"Target tokenizer BOS: \"{self._start_token}\" and EOS: \"{self._end_token}\"")

        # TSV delimiter
        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = source_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.quoting = quoting
        self._maybe_translate_sample_factor = maybe_translate_sample_factor

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
                # First case is the NL\tLOCALE case
                if len(row) == 4:
                    source_sequence, source_lang, target_sequence, target_lang = row
                else:
                    raise ConfigurationError(
                        "Invalid line format: %s (line number %d)" % (row, line_num + 1)
                    )

                yield self.text_to_instance(source_string=source_sequence, target_string=target_sequence,
                                            source_lang=source_lang, target_lang=target_lang)

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
            self, source_string: str, target_string: str = None, source_lang: str = None, target_lang: str = None
    ) -> Instance:  # type: ignore
        source_lang = self._validator(source_lang)
        tokenizer_args = (source_string, source_lang) if self._mbart50_tokenizer else (source_string, )
        if self._mbart_tokenizer:
            self._source_tokenizer.tokenizer.src_lang = source_lang
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

        # Passing in target_string as a copy of the source is not redundant as we need different BOS/EOS behaviour
        if target_string is not None:
            # If we are above threshold then X->X, else X->EN translation objective
            if random() > self._maybe_translate_sample_factor:
                target_string = source_string
                target_lang = source_lang

            tokenizer_args = (target_string, target_lang, False) if self._mbart50_tokenizer else (target_string,)
            tokenized_target = self._target_tokenizer.tokenize(*tokenizer_args)

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
