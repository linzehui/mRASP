# Pre-training Multilingual Neural Machine Translation by Leveraging Alignment Information, EMNLP2020

This is the repo for EMNLP2020 paper Pre-training Multilingual Neural Machine Translation by Leveraging Alignment Information.

[[paper](https://arxiv.org/abs/2010.03142)]

<img src="https://github.com/linzehui/mRASP/blob/master/logo.png" width="30%" height="50%">


## News
We have evolved our mRASP into mRASP2/mCOLT, which is a much stronger many-to-many multilingual model. mRASP2 has been accepted by ACL2021 main conference. Welcome to use mRASP2.

* [[paper](https://arxiv.org/abs/2105.09501)]
* [[code](https://github.com/PANXiao1994/mRASP2)]

## Introduction

mRASP, representing multilingual Random Aligned Substitution Pre-training, is a pre-trained multilingual neural machine translation model. mRASP is pre-trained on large scale multilingual corpus containing 32 language pairs. The obtained model can be further ﬁne-tuned on downstream language pairs. To effectively bring words and phrases with similar meaning closer in representation across multiple languages, we introduce Random Aligned Substitution (RAS) technique. Extensive experiments conducted on different scenarios demonstrate the efficacy of mRASP. For detailed information please refer to the paper.  

## Structure
```text
.
├── experiments                             # Example files: including configs and data
├── preprocess                              # The preprocess step
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── common.sh           
│   │   ├── data_preprocess/                # clean + tokenize
│   │   │   ├── __init__.py
│   │   │   ├── clean_scripts/
│   │   │   ├── tokenize_scripts/
│   │   │   ├── clean_each.sh
│   │   │   ├── prep_each.sh
│   │   │   ├── prep_mono.sh                # preprocess a monolingual corpus
│   │   │   ├── prep_parallel.sh            # preprocess a parallel corpus
│   │   │   └── tokenize_each.sh
│   │   ├── misc/
│   │   │   ├── __init__.py
│   │   │   ├── multilingual_preprocess_yml_generator.py
│   │   │   └── multiprocess.sh
│   │   ├── ras/
│   │   │   ├── __init__.py
│   │   │   ├── random_alignment_substitution.sh
│   │   │   ├── random_alignment_substitution_w_multi.sh 
│   │   │   ├── replace_word.py  # RAS using MUSE bilingual dict
│   │   │   └── replace_word_w_multi.py  # RAS using multi-way parallel dict
│   │   └── subword/
│   │       ├── __init__.py
│   │       ├── multilingual_apply_subword_vocab.sh     # script to only apply subword (w/o learning new vocab)
│   │       ├── multilingual_learn_apply_subword_vocab_joint.sh     # script to learn new vocab and apply subword
│   │       └── scripts/
│   ├── __init__.py
│   ├── multilingual_merge.sh               # script to merge multiple parallel dataset
│   ├── multilingual_preprocess_main.sh     # main entry for preprocess
│   └── README.md    
├── train                        
│   ├── __init__.py
│   ├── misc/
│   │   ├── load_config.sh
│   │   └── monitor.sh                  # script to monitor the generation of checkpoint and evaluate them
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── average_checkpoints_from_file.py
│   │   ├── average_ckpt.sh             # checkpoint average
│   │   ├── common_scripts.sh
│   │   ├── get_worst_ckpt.py
│   │   ├── keep_top_ckpt.py
│   │   ├── remove_bpe.py
│   │   └── rerank_utils.py
│   ├── pre-train.sh                    # main entry for pre-train
│   ├── fine-tune.sh                    # main entry for fine-tune
│   └── README.md
├── requirements.txt
└── README.md
```

## Pre-requisite
```bash
pip install -r requirements.txt
```

## Pipeline
The pipeline contains two steps: Pre-train and Fine-tune. We first pre-train our model on multiple language pairs jointly. Then we further fine-tune on downstream language pairs.

### Preprocess
The preprocess pipeline is composed of the following 4 separate steps:

* Data filtering and cleaning

* Tokenization

* Learn / Apply joint bpe sub-word vocabulary

* Random Alignment Substitution (optional, only valid for train set)

We provide a script to run all the above steps in one command:
```bash
bash ${PROJECT_ROOT}/preprocess/multilingual_preprocess_main.sh ${config_yaml_file}
```

### Pre-train

step1: preprocess train data and learn a joint BPE subword vocabulary across all languages.
```bash
bash ${PROJECT_ROOT}/preprocess/multilingual_preprocess_main.sh ${PROJECT_ROOT}/experiments/example/configs/preprocess/train.yml
```
The command above will do clean, subword, merge, ras, step by step. Now we have a BPE vocabulary and an RASed multilingual dataset merged from multiple language pairs.

step2: preprocess development data
```bash
bash ${PROJECT_ROOT}/preprocess/multilingual_preprocess_main.sh ${PROJECT_ROOT}/experiments/example/configs/preprocess/dev.yml
```
We create a multilingual development set to help choose the best pre-trained checkpoint.

step3: binarize data
```bash
bash ${PROJECT_ROOT}/experiments/example/bin_pretrain.sh
```

step4: pre-train on RASed multilingual corpus
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 && bash ${PROJECT_ROOT}/train/pre-train.sh ${PROJECT_ROOT}/experiments/example/configs/train/pre-train/transformer_big.yml
```
You can modify the configs to choose the model architecture or dataset used.

### Fine-tune

step1: preprocess train/test data
```bash
bash ${PROJECT_ROOT}/preprocess/multilingual_preprocess_main.sh ${PROJECT_ROOT}/experiments/example/configs/preprocess/train_en2de.yml
bash ${PROJECT_ROOT}/preprocess/multilingual_preprocess_main.sh ${PROJECT_ROOT}/experiments/example/configs/preprocess/test_en2de.yml
```
The command above will do: clean and subword.

step2: binarize data
```bash
bash ${PROJECT_ROOT}/experiments/example/bin_finetune.sh
```

step3: fine-tune on specific language pairs
```bash
export CUDA_VISIBLE_DEVICES=0,1,2 && export EVAL_GPU_INDEX=${eval_gpu_index} && bash ${PROJECT_ROOT}/train/fine-tune.sh ${PROJECT_ROOT}/experiments/example/configs/train/fine-tune/en2de_transformer_big.yml ${PROJECT_ROOT}/experiments/example/configs/eval/en2de_eval.yml
```
* `eval_gpu_index` denotes the index of gpu on your machine that will be allocated to evaluate the model. if you set it to `-1`, it means that cpu will be used for evaluating during training.


## Multilingual Pre-trained Model

### Dataset

We merge 32 English-centric language pairs, resulting in 64 directed translation pairs in total. The original 32 language pairs corpus contains about 197M pairs of sentences. We get about 262M pairs of sentences after applying RAS, since we keep both the original sentences and the substituted sentences. We release both the original dataset and dataset after applying RAS. (Note that if you can't download the files, please replace the download link prefix "sf3-ttcdn-tos.pstatp.com" with "lf3-nlp-opensource.bytetos.com".)

| Dataset | #Pair |
| --- | --- |
| [32-lang-pairs-TRAIN](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/pretrain/dataset/train/32-lang-pairs/download.sh) | 197603294 |
| [32-lang-pairs-RAS-TRAIN](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/pretrain/dataset/train/32-lang-pairs-RAS/download.sh) | 262662792 |
| [32-lang-pairs-DEV](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/pretrain/dataset/dev/download.sh) | 156587 |
| [Vocab](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/pretrain/dataset/vocab.bpe.32000) | - |
| [BPE Code](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/pretrain/dataset/codes.bpe.32000) | - |


### Checkpoints

We release checkpoints trained on 32-lang-pairs and 32-lang-pairs-RAS. We also extend our model to 58 language pairs.

| Dataset | Checkpoint |
| --- | --- |
| Baseline-w/o-RAS | [mTransformer-6enc6dec](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/pretrain/checkpoint/mTransformer-6enc6dec.pt) |
| mRASP-PC32 | [mRASP-PC32-6enc6dec](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/pretrain/checkpoint/mRASP-PC32-6enc6dec.pt) |
| mRASP-PC58 | - |

## Fine-tuning Model

We release En-Ro, En2De and En2Fr benchmark checkpoints and the corresponding configs.


| Lang-Pair | Datasource | Checkpoints | Configs | tok-BLEU | detok-BLEU |
| --- | --- | --- | --- | --- | --- |
| En2Ro | [WMT16 En-Ro](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-ro/dataset/train/download.sh), [dev](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-ro/dataset/dev/download.sh), [test](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-ro/dataset/test/download.sh) | [en2ro](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-ro/checkpoint/en2ro_checkpoint.pt) | [en2ro_config](experiments/fine-tune-configs/en2ro_config.yml) | 39.0 | 37.6 |
| Ro2En | [WMT16 Ro-En](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/ro-en/dataset/train/download.sh), [dev](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/ro-en/dataset/dev/download.sh), [test](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/ro-en/dataset/test/download.sh) | [ro2en](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-ro/checkpoint/ro2en_checkpoint.pt) | [ro2en_config](experiments/fine-tune-configs/ro2en_config.yml) | 37.7 | 36.9 |
| En2De | [WMT16 En-De](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-de/dataset/train/download.sh), [newstest16](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-de/dataset/newstest16/download.sh) | [en2de](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-de/checkpoint/en2de_checkpoint.pt) | [en2de_config](experiments/fine-tune-configs/en2de_config.yml) | 30.3 | - |
| En2Fr | [WMT14 En-Fr](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-fr/dataset/train/download.sh), [newstest14](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-fr/dataset/newstest14/download.sh) | [en2fr](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-fr/checkpoint/en2fr_checkpoint.pt) | [en2fr_config](experiments/fine-tune-configs/en2fr_config.yml) | 44.3 | - |



## Comparison with mBART

mBART is a pre-trained model trained on large-scale multilingual corpora. To illustrate the superiority of mRASP, we also compare our results with mBART. We choose different scales of language pairs and use the same test sets as mBART.

| Lang-pairs | Size | Direction | Datasource | Testset | Checkpoint | mBART | mRASP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| En-Gu | 10K | ⟶ | [en_gu_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-gu/train/download.sh) | [newstest19](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-gu/en2gu-newstest19/download.sh) | [en2gu](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-gu/checkpoint/en2gu_checkpoint.pt) | 0.1 | **3.2** |
|  |  | ⟵ | [en_gu_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-gu/train/download.sh) | [newstest19](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-gu/gu2en-newstest19/download.sh) | [gu2en](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-gu/checkpoint/gu2en_checkpoint.pt) | 0.3 | **0.6** |
| En-Kk | 128K | ⟶ | [en_kk_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-kk/train/download.sh) | [newstest19](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-kk/en2kk-newstest19/download.sh) | [en2kk](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-kk/checkpoint/en2kk_checkpoint.pt) | 2.5 | **8.2** |
|  |  | ⟵ | [en_kk_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-kk/train/download.sh) | [newstest19](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-kk/kk2en-newstest19/download.sh) | [kk2en](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-kk/checkpoint/kk2en_checkpoint.pt) | 7.4 | **12.3** |
| En-Tr | 388K | ⟶ | [en_tr_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-tr/train/download.sh) | [newstest17](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-tr/en2tr-newstest17/download.sh) | [en2tr](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-tr/checkpoint/en2tr_checkpoint.pt) | 17.8 | **20.0** |
|  | | ⟵ | [en_tr_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-tr/train/download.sh) | [newstest17](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-tr/tr2en-newstest17/download.sh) | [tr2en](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-tr/checkpoint/tr2en_checkpoint.pt) | 22.5 | **23.4** |
| En-Et | 2.3M | ⟶ | [en_et_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-et/train/download.sh) | [newstest18](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-et/newstest18-en2et/download.sh) | [en2et](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-et/checkpoint/en2et_checkpoint.pt) | **21.4** | 20.9 |
|  |  | ⟵ | [en_et_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-et/train/download.sh) | [newstest18](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-et/newstest18-et2en/download.sh) | [et2en](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-et/checkpoint/et2en_checkpoint.pt) | **27.8** | 26.8 |
| En-Fi | 4M | ⟶ | [en_fi_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-fi/train/download.sh) | [newstest17](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-fi/en2fi-newstest17/download.sh) | [en2fi](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-fi/checkpoint/en2fi_checkpoint.pt) | 22.4 | **24.0** |
|  |  | ⟵ | [en_fi_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-fi/train/download.sh) | [newstest17](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-fi/fi2en-newstest17/download.sh) | [fi2en](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-fi/checkpoint/fi2en_checkpoint.pt) | **28.5** | 28.0 |
| En-Lv | 5.5M | ⟶ | [en_lv_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-lv/train/download.sh) | [newstest17](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-lv/en2lv-newstest17/download.sh) | [en2lv](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-lv/checkpoint/en2lv_checkpoint.pt) | 15.9 | **21.6** |
|  |  | ⟵ | [en_lv_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-lv/train/download.sh) | [newstest17](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-lv/lv2en-newstest17/download.sh) | [lv2en](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-lv/checkpoint/lv2en_checkpoint.pt) | 19.3 | **24.4** |
| En-Cs | 978K | ⟶ | [en_cs_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-cs/train/download.sh) | [newstest19](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-cs/newstest19/download.sh) | [en2cs](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-cs/checkpoint/en2cs_checkpoint.pt) | 18.0 | **19.9** |
| En-De | 4.5M | ⟶ | [en_de_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-de/dataset/train/download.sh) | [newstest19](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/compare-with-mbart/en-de/newstest19/download.sh) | [en2de](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-de/checkpoint/en2de_checkpoint.pt) |  30.5 | **35.2** |
| En-Fr | 40M | ⟶ | [en_fr_train](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-fr/dataset/train/download.sh) | [newstest14](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-fr/dataset/newstest14/download.sh) | [en2fr](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/emnlp2020/mrasp/en-fr/checkpoint/en2fr_checkpoint.pt) | 41.0 | **44.3** |

## Citation
If you are interested in mRASP, please consider citing our paper:
```
@inproceedings{lin-etal-2020-pre,
    title = "Pre-training Multilingual Neural Machine Translation by Leveraging Alignment Information",
    author = "Lin, Zehui  and
      Pan, Xiao  and
      Wang, Mingxuan  and
      Qiu, Xipeng  and
      Feng, Jiangtao  and
      Zhou, Hao  and
      Li, Lei",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.210",
    pages = "2649--2663",
}
```

