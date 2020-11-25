# Pre-training Multilingual Neural Machine Translation by Leveraging Alignment Information, EMNLP2020

This is the repo for EMNLP2020 paper Pre-training Multilingual Neural Machine Translation by Leveraging Alignment Information.

[[paper](https://arxiv.org/abs/2010.03142)]

<img src="https://github.com/linzehui/mRASP/blob/master/logo.png" width="30%" height="50%">


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

We merge 32 English-centric language pairs, resulting in 64 directed translation pairs in total. The original 32 language pairs corpus contains about 197M pairs of sentences. We get about 262M pairs of sentences after applying RAS, since we keep both the original sentences and the substituted sentences. We release both the original dataset and dataset after applying RAS.

| Dataset | #Pair |
| --- | --- |
| [32-lang-pairs-TRAIN](https://www.icloud.com/iclouddrive/0BEI1y-fS10HCCugKiYdawB6w#32-lang-pairs) | 197603294 |
| [32-lang-pairs-RAS-TRAIN](https://www.icloud.com/iclouddrive/0S3n2GVIfBoTvtt2f0FxW1gIg#32-lang-pairs-RAS) | 262662792 |
| [32-lang-pairs-DEV](https://www.icloud.com/iclouddrive/0gnMc_nc1QJfzxdOAIzfEMszA#dev) | 156587 |
| [Vocab](https://www.icloud.com/iclouddrive/0WF9bJk0-m9hPErUNjWabHzow#vocab.bpe.32000) | - |
| [BPE Code](https://www.icloud.com/iclouddrive/0v5NQ7XgGi1fn0y8X0GiVs4EA#codes.bpe.32000) | - |


### Checkpoints

We release checkpoints trained on 32-lang-pairs and 32-lang-pairs-RAS. We also extend our model to 58 language pairs.

| Dataset | Checkpoint |
| --- | --- |
| 32-lang-pairs | [32-lang-pairs-ckp](https://www.icloud.com/iclouddrive/0C7D5K7D-QkX3E77rGdBwt-JQ#pretrain%5Fcheckpoint%5Flast%5Fwithout%5FRAS) |
| 32-lang-pairs-RAS | [32-lang-pairs-RAS-ckp](https://www.icloud.com/iclouddrive/0qOUbmoRIUYEkIjEQ9TGyV6QQ#pretrain%5Fcheckpoint%5Flast%5FRAS) |
| 58-lang-pairs-RAS | - |

## Fine-tuning Model

We release En-Ro, En2De and En2Fr benchmark checkpoints and the corresponding configs.


| Lang-Pair | Datasource | Checkpoints | Configs | tok-BLEU | detok-BLEU |
| --- | --- | --- | --- | --- | --- |
| En2Ro | [WMT16 En-Ro](https://www.icloud.com/iclouddrive/0NOsmt2XINO2jPco1eI4fSaYA#dataset) | [en2ro](https://www.icloud.com/iclouddrive/02sQRe56KZAOsmnKkQUlosCvA#checkpoint) | [en2ro_config](experiments/fine-tune-configs/en2ro_config.yml) | 39.0 | 37.6 |
| Ro2En | [WMT16 Ro-En](https://www.icloud.com/iclouddrive/0WSui8ndtwp1-8aSXJuKc7pLQ#dataset) | [ro2en](https://www.icloud.com/iclouddrive/0E56mveWMJrcXKves_2JNezjw#checkpoint) | [ro2en_config](experiments/fine-tune-configs/ro2en_config.yml) | 37.7 | 36.9 |
| En2De | [WMT16 En-De](https://www.icloud.com/iclouddrive/0wBL1hyZ5wp_AQ2xe0J3hEuew#dataset) | [en2de](https://www.icloud.com/iclouddrive/0cnfeN1qZWVzJ4Oaauz_K3tVQ#checkpoint) | [en2de_config](experiments/fine-tune-configs/en2de_config.yml) | 30.3 | - |
| En2Fr | [WMT14 En-Fr](https://www.icloud.com/iclouddrive/08SAsEeoEtVSc02kE7lgrDeMg#dataset) | [en2fr](https://www.icloud.com/iclouddrive/0-bzvB4PMmy7gFJVg24pL1hzQ#checkpoint) | [en2fr_config](experiments/fine-tune-configs/en2fr_config.yml) | 44.3 | - |



## Comparison with mBART

mBART is a pre-trained model trained on large-scale multilingual corpora. To illustrate the superiority of mRASP, we also compare our results with mBART. We choose different scales of language pairs and use the same test sets as mBART.

| Lang-pairs | Size | Direction | Datasource | Testset | Checkpoint | mBART | mRASP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| En-Gu | 10K | ⟶ | [en_gu_train](https://www.icloud.com/iclouddrive/0_xCBSPuTgq6o31vi-AhzKOAw#train) | [newstest19](https://www.icloud.com/iclouddrive/00tXKTYSc93H_IbUWkFWusxSw#en2gu-newstest19) | [en2gu](https://www.icloud.com/iclouddrive/0ZOzhYC7JzMQFnZQ41SW-_BEw#checkpoint) | 0.1 | **3.2** |
|  |  | ⟵ | [en_gu_train](https://www.icloud.com/iclouddrive/0_xCBSPuTgq6o31vi-AhzKOAw#train) | [newstest19](https://www.icloud.com/iclouddrive/0pmckSz7NfUhluwjNW9MdImDA#gu2en-newstest19) | [gu2en](https://www.icloud.com/iclouddrive/0ZOzhYC7JzMQFnZQ41SW-_BEw#checkpoint) | 0.3 | **0.6** |
| En-Kk | 128K | ⟶ | [en_kk_train](https://www.icloud.com/iclouddrive/0NEfLI1Po4-q-OORLCtUyZDuw#train) | [newstest19](https://www.icloud.com/iclouddrive/02Qj7FFJc9PHqMdN_Uk2cMR5A#en2kk-newstest19) | [en2kk](https://www.icloud.com/iclouddrive/0mO6_Ozylj3H7Pm2SWnolTsPg#checkpoint) | 2.5 | **8.2** |
|  |  | ⟵ | [en_kk_train](https://www.icloud.com/iclouddrive/0NEfLI1Po4-q-OORLCtUyZDuw#train) | [newstest19](https://www.icloud.com/iclouddrive/0rX7j3-nuymvJJGDrPEaJYwaw#kk2en-newstest19) | [kk2en](https://www.icloud.com/iclouddrive/0mO6_Ozylj3H7Pm2SWnolTsPg#checkpoint) | 7.4 | **12.3** |
| En-Tr | 388K | ⟶ | [en_tr_train](https://www.icloud.com/iclouddrive/0xnEm2yG6s4stakbQCAVL3UnA#train) | [newstest17](https://www.icloud.com/iclouddrive/0mujfGmF14jcvxTUpqP384c7A#en2tr-newstest17) | [en2tr](https://www.icloud.com/iclouddrive/0_-EbFpGsG-AZBWSdHBNTIh6Q#checkpoint) | 17.8 | **20.0** |
|  | | ⟵ | [en_tr_train](https://www.icloud.com/iclouddrive/0xnEm2yG6s4stakbQCAVL3UnA#train) | [newstest17](https://www.icloud.com/iclouddrive/0GMOj6rpH9NHcTK-6dNtU5MrA#tr2en-newstest17) | [tr2en](https://www.icloud.com/iclouddrive/0_-EbFpGsG-AZBWSdHBNTIh6Q#checkpoint) | 22.5 | **23.4** |
| En-Et | 2.3M | ⟶ | [en_et_train](https://www.icloud.com/iclouddrive/0COCD6KMaPjtr6R8GPE4whMVw#train) | [newstest18](https://www.icloud.com/iclouddrive/0fz8Y2CzrNODaVZPV5JDv1Apg#newstest18-en2et) | [en2et](https://www.icloud.com/iclouddrive/0aEH-5xiQ-xzWgPfa8nEEFyvg#checkpoint) | **21.4** | 20.9 |
|  |  | ⟵ | [en_et_train](https://www.icloud.com/iclouddrive/0COCD6KMaPjtr6R8GPE4whMVw#train) | [newstest18](https://www.icloud.com/iclouddrive/0hyJ-jOM8P4qHg6S6rw3Uo5Nw#newstest18-et2en) | [et2en](https://www.icloud.com/iclouddrive/0aEH-5xiQ-xzWgPfa8nEEFyvg#checkpoint) | **27.8** | 26.8 |
| En-Fi | 4M | ⟶ | [en_fi_train](https://www.icloud.com/iclouddrive/0cDUcgvGOH7GnwC3tMmHM7F-w#train) | [newstest17](https://www.icloud.com/iclouddrive/0hQfdK5rTDMtoYnu2K4UPAwtg#en2fi-newstest17) | [en2fi](https://www.icloud.com/iclouddrive/0qrFD5j4zkwVy6_U4VbPTfa9w#checkpoint) | 22.4 | **24.0** |
|  |  | ⟵ | [en_fi_train](https://www.icloud.com/iclouddrive/0cDUcgvGOH7GnwC3tMmHM7F-w#train) | [newstest17](https://www.icloud.com/iclouddrive/0EtCeRYPeJWsCwV51Lpiyd-eQ#fi2en-newstest17) | [fi2en](https://www.icloud.com/iclouddrive/0qrFD5j4zkwVy6_U4VbPTfa9w#checkpoint) | **28.5** | 28.0 |
| En-Lv | 5.5M | ⟶ | [en_lv_train](https://www.icloud.com/iclouddrive/05jv7Hkq_Mpl8vLq9PxK0opZA#train) | [newstest17](https://www.icloud.com/iclouddrive/0rBB7u0x6l-g3-z7kKAnCt9Rw#en2lv-newstest17) | [en2lv](https://www.icloud.com/iclouddrive/0qPgq4hyB5c-xY3QANyufmLNg#checkpoint) | 15.9 | **21.6** |
|  |  | ⟵ | [en_lv_train](https://www.icloud.com/iclouddrive/05jv7Hkq_Mpl8vLq9PxK0opZA#train) | [newstest17](https://www.icloud.com/iclouddrive/06y62k98NmumwYtEctbuLzozg#lv2en-newstest17) | [lv2en](https://www.icloud.com/iclouddrive/0qPgq4hyB5c-xY3QANyufmLNg#checkpoint) | 19.3 | **24.4** |
| En-Cs | 978K | ⟶ | [en_cs_train](https://www.icloud.com/iclouddrive/0gSrgAUuCIwqI0X-_SMx0mrAQ#train) | [newstest19](https://www.icloud.com/iclouddrive/0oyjkTYnlXXikmyR82G6RYB2w#newstest19) | [en2cs](https://www.icloud.com/iclouddrive/0CmG9bFBhcKVrEuxCvIKsFiCA#checkpoint) | 18.0 | **19.9** |
| En-De | 4.5M | ⟶ | [en_de_train](https://www.icloud.com/iclouddrive/0wBL1hyZ5wp_AQ2xe0J3hEuew#dataset) | [newstest19](https://www.icloud.com/iclouddrive/0jwrbrN6_mHjI0RrDkrA9kc_g#newstest19) | [en2de](https://www.icloud.com/iclouddrive/0cnfeN1qZWVzJ4Oaauz_K3tVQ#checkpoint) |  30.5 | **35.2** |
| En-Fr | 40M | ⟶ | [en_fr_train](https://www.icloud.com/iclouddrive/08SAsEeoEtVSc02kE7lgrDeMg#dataset) | [newstest14](https://www.icloud.com/iclouddrive/0tHOKNsMgTu-3szmw6bHQlViw#newstest14) | [en2fr](https://www.icloud.com/iclouddrive/0-bzvB4PMmy7gFJVg24pL1hzQ#checkpoint) | 41.0 | **44.3** |

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

