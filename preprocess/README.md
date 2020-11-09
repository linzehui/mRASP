# Preparing the data
## Pre-requisite
First, get the data, including parallel data (train and dev) and auxiliary monolingual data (for learning a joint bpe sub-word vocabulary);


Then install subword-nmt and tokenizers (including MosesTokenier, Kytea, and etc. for different languages)
```bash
pip install subword-nmt
pip install mosestokenizer
pip install kytea
```

## The preprocess pipeline

The preprocess pipeline is composed of the following 4 separate steps:

* Data filtering and cleaning

* Tokenization

* Learn / Apply joint bpe sub-word vocabulary

* Random Alignment Substitution (optional, only valid for train set)

We provide a script to run all the above steps in one command:
```bash
bash ${PROJECT_ROOT}/preprocess/multilingual_preprocess_main.sh ${config_yaml_file}
```

### Example
create a yaml config file: 
```text
file_prefix: train
raw_data_path: ${PROJECT_ROOT}/experiments/toy/data/raw/train
merged_output_path: ${PROJECT_ROOT}/experiments/toy/merged_data
output_main_path: ${PROJECT_ROOT}/experiments/toy/data/prep/train
configs_subdir: configs
cleaned_subdir: cleaned
logs_subdir: logs
vocab_subdir: vocab
output_subdir: output
final_vocab_path: ${PROJECT_ROOT}/experiments/toy/vocab
preprocess_steps_list: clean:subword:merge:ras
learn: false
mono: false
mono_resample: /opt/tiger/mrasp/data/ratio.yml
subword_bpe_merge_ops: 600
maximum_vocabulary_size: 1000
minimum_frequency: 5
pairs: en_de;en_cs
directions: en2de;de2en;en2cs;cs2en
languages: en;de;cs
default_pairs:
  deduplicate: true
  keep_lines_percent: '1.0'
  shuffle: true
default_langs:
  do_normalize_punctuations: true
  tokenizer: MosesTokenizer
ras:
  dict_path: ${PROJECT_ROOT}/experiments/toy/dictionaries
  vocab_size: 1000
  max_dep: 3
  multi_dict_path: /opt/tiger/mrasp/data/dict/dict.merge_dep3.txt
  languages: zh;en;bg
  target_languages: en;af;ar;be;bg;cs;de;el;eo;es;et;fi;fr;gu;he;hi;it;ja;ka;kk;ko;lt;lv;mn;ms;mt;my;ro;ru;sr;tr;vi;zh
```
* `file_prefix` denotes the type of the file (`train` or `dev`)
* `raw_data_path` denotes the path of raw data, the directory tree structure must strictly follow the example:

```text
1. for `file_prefix=train`
├── cs/
│   └── train.cs
├── de/
│   └── train.de
├── en/
│   └── train.en
├── en_cs/
│   ├── train.cs
│   └── train.en
└── en_de/
    ├── train.de
    └── train.en
    
2. for `file_prefix=dev`
├── en2cs/
│   ├── dev.cs
│   └── dev.en
├── cs2en/
│   ├── dev.cs
│   └── dev.en
├── en2de/
│   ├── dev.de
│   └── dev.en
├── de2en/
│   ├── dev.de
│   └── dev.en
├── en2fr/
│   ├── dev.fr
│   └── dev.en
└── fr2en/
    ├── dev.fr
    └── dev.en

```
* `merged_output_path`: the final output path after all preprocess procedures.
* `output_main_path`: the root path of the preprocessed data
* `configs_subdir`: the subdir under `${output_main_path}` that stores generated config files during preprocess
* `cleaned_subdir`: the subdir under `${output_main_path}` that stores cleaned files during preprocess
* `logs_subdir`: the subdir under `${output_main_path}` that stores logs for each step during preprocess
* `output_subdir`: the subdir under `${output_main_path}` that stores files after BPE sub-word during preprocess
* `vocab_subdir`: the subdir under `${output_main_path}` that stores generated vocabularies during preprocess
```text
 ├── cleaned/
 │   ├── cs/
 │   ├── de/
 │   ├── en/
 │   ├── en_cs/
 │   └── en_de/
 ├── configs/
 ├── logs/
 └── output/
    ├── en_cs/
    └── en_de/
```
* `final_vocab_path`: the final vocabulary path after appending language tokens to the end.
* `preprocess_steps_list`: the preprocess steps involved in one pass, split by ':'. Normally you should run all 4
 steps together. Only when the output file of the previous step already exists, the next step can be executed. That means,
 "clean:merge" is illegal; "subword:merge" can only be properly executed if "clean" is already previously executed. Additionally, the step `ras` is only valid for training set.
* `mono` denotes the data is monolingual or not. There is slight difference for monolingual data and parallel data. `mono_resample` contains yaml file specifying the sampling ratio of each language, if not provided, no resample procedure will be conducted.
* `ras_multi_dict_path` denotes the multi-way parallel dict, if provided, use the provided multi-way parallel dict, otherwise use MUSE bilingual dict by default.
* `ras_max_depth` and `ras_languages` are only valid when `ras_multi_dict_path` is set. `ras_languages` means the languages that can be replaced to other languages, if not set, use `languages` by default. `ras_max_depth` means the maximum valid depth of the multi-way dict.
* `ras_target_languages` means valid target languages of RAS procedure.

## Binarize
Run the command
```bash
fairseq-preprocess \
        --source-lang src --target-lang trg \
        --srcdict ${final_vocab_path}/vocab.bpe.${subword_bpe_merge_ops} \
        --tgtdict ${final_vocab_path}/vocab.bpe.${subword_bpe_merge_ops} \
        --trainpref ${merged_output_path}/train \
        --validpref ${merged_output_path}/dev \
        --destdir ${binarized_data_path} \
        --workers ${num_cpus}
```


