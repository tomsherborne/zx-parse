# [Zero-Shot Cross-Lingual Semantic Parsing](https://arxiv.org/abs/2104.07554)
_Proceedings of ACL 2022, Dublin Ireland_

###### Abstract
Recent work in cross-lingual semantic parsing has successfully applied machine translation to localize parsers to new languages. However, these advances assume access to high-quality machine translation systems and word alignment tools. We remove these assumptions and study cross-lingual semantic parsing as a zero-shot problem, without parallel data (i.e., utterance-logical form pairs) for new languages. We propose a multi-task encoder-decoder model to transfer parsing knowledge to additional languages using only English-logical form paired data and in-domain natural language corpora in each new language. Our model encourages language-agnostic encodings by jointly optimizing for logical-form generation with auxiliary objectives designed for cross-lingual latent representation alignment. Our parser performs significantly above translation-based baselines and, in some cases, competes with the supervised upper-bound. 

Our model is called __ZX-Parse__ for a **Z**ero-Shot **Cross**-lingual Semantic **Parse**r. Also as a tribute to the [ZX Spectrum](https://www.nme.com/features/gaming-features/zx-spectrum-at-40-a-look-back-3162913). 

## Installation

Install a new Conda environment:
```
conda env create --name zxp --file=zxparse.yaml
```
You may need to install [AllenNLP](https://github.com/allenai/allennlp), [AllenNLP Models](https://github.com/allenai/allennlp-models) and [HuggingFace Transformers](https://github.com/huggingface/transformers) from source.

### Data

__ATIS__: we create a new version of ATIS combining executable SQL queries with the utterances in six languages from [MultiATIS++](https://github.com/amazon-research/multiatis). The languages are English, French, Portuguese, Spanish, German and Chinese. 

We refer to this split as __MultiATIS++SQL__ and is available in the [bootstrap_atis](https://github.com/tomsherborne/bootstrap_atis) repository. This dataset is subject to LDC Licencing requirements. If you require access to this then [email me](mailto:tom.sherborne@ed.ac.uk). We will ask for evidence of holding relevant LDC licenses (LDC93S5, LDC94S19, and LDC95S26).

[__Overnight__:](https://nlp.stanford.edu/pubs/wang-berant-liang-acl2015.pdf) multi-domain semantic parsing dataset in English with German and Chinese test data is available [here](https://github.com/tomsherborne/bootstrap) in the repository for our previous paper [Bootstrapping a Crosslingual Semantic Parser](https://aclanthology.org/2020.findings-emnlp.45/).

__MKQA__: the ZX-Parse model uses natural language corpora (text without any logical form pairing) to promote cross-lingual latent similarity. We use [MKQA V1](https://github.com/apple/ml-mkqa) for this as it has both _broad language coverage_ and _models native speaker questions_. We upload the sample we use here. We also do experiments on [ParaCrawl](https://aclanthology.org/2020.acl-main.417/), but find that using _questions_ is superior to using _arbitrary declarative text_.

When data is downloaded then copy this into the `./data/` folder. You will also need to install relevant knowledge bases. For __ATIS__, you need to install the ATIS SQL database. We provide an interface to execute queries in an assumed MySQL database in `./exec/`. For Overnight, you will need the relevant version of [SEMPRE](https://nlp.stanford.edu/software/sempre/). To avoid replicating SEMPRE, we do not provide this interface but if you want help getting this running then [email me](mailto:tom.sherborne@ed.ac.uk).

### Source Code

The core model used in the paper is `seq2seq_multihead_lambda.py`. Everything else is used to make this model run (or older/ablation models.
```
zxp
├── dataset_readers
│   ├── __init__.py
│   ├── seq2seq_pretrain_bitext.py
│   ├── seq2seq_pretrain_paired.py
│   └── seq2seq_pretrain_unpaired.py
├── initializers
│   ├── initializers.py
│   └── __init__.py
├── __init__.py
├── metrics
│   ├── __init__.py
│   └── token_sequence_accuracy.py
├── models
│   ├── composed_seq2seq_kwargs.py
│   ├── __init__.py
│   ├── seq2seq_multihead_lambda.py
│   ├── seq2seq_multihead.py
│   └── seq2seq_singlehead_lambda.py
├── modules
│   ├── auto_regressive_mod.py
│   ├── gradient_reversal_layer.py
│   ├── hf_transformer_utils.py
│   ├── __init__.py
│   └── transformer_encoder.py
├── predictors
│   ├── __init__.py
│   └── seq2seq_anyhead.py
└── trainers
    ├── gradient_descent_lambda.py
    └── __init__.py

```

### Config Files

The complete AllenNLP configuration file for the best model is in `config/mheadsql_arr_atis_paracrawl_bt_factor_0.5.json`. To run this:

```
export SERIAL_DIR=${PWD}/experiments/zxp_atis_best
export EXP_CONFIG=${PWD}/config/mheadsql_arr_atis_paracrawl_bt_factor_0.5.json
mkdir -p ${SERIAL_DIR}
allennlp train "${EXP_CONFIG}" \
    --serialization-dir "${SERIAL_DIR}" \
    --include-package zxp \
    --force
```

To predict:

```
export SERIAL_DIR=${PWD}/experiments/zxp_atis_best
export JSON_PRED_INPUT=/path/to/jsonlines/test/inputs
export JSON_PRED_OUTPUT=${SERIAL_DIR}/test_output_locale_XX.json
export TOK_PRED_OUTPUT=${SERIAL_DIR}/test_output_locale_XX.txt

allennlp predict "${SERIAL_DIR}/model.tar.gz" \
                 "${JSON_PRED_INPUT}" \
                 --cuda-device 0 \
                 --output-file "${JSON_PRED_OUTPUT}" \
                 --include-package pretrain_code \
                 --predictor "seq2seq_anyhead" \
                 --overrides "{\"model.decoder_sql.beam_size\": 5}" 

python "${EXEC_ROOT}/extract_tokens.py" "${JSON_PRED_OUTPUT}" "${TOK_PRED_OUTPUT}"
```

To execute predictions (assuming ATIS):
```
export TOK_PRED_OUTPUT=${SERIAL_DIR}/test_output_locale_XX.txt
export DNT_PRED_OUTPUT=${SERIAL_DIR}/test_output_locale_XX.denot
bash ${PWD}/exec/sql/execute_sql.sh ${TOK_PRED_OUTPUT} ${DNT_PRED_OUTPUT}
```

### Citation
If you use our code or new data split for ATIS then cite our paper (please cross-check the citation in the ACL Anthology):
```
@inproceedings{sherborne-lapata-2022-zero-shot,
    title = "Zero-Shot Cross-lingual Semantic Parsing",
    author = "Sherborne, Tom  and
      Lapata, Mirella",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics"
}
```
If you use this split of Overnight with German and Chinese data then cite our previous paper:
```
@inproceedings{sherborne-etal-2020-bootstrapping,
    title = "Bootstrapping a Crosslingual Semantic Parser",
    author = "Sherborne, Tom  and
      Xu, Yumo  and
      Lapata, Mirella",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.45",
    doi = "10.18653/v1/2020.findings-emnlp.45",
    pages = "499--517"
}
```