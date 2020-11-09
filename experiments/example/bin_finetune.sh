#!/usr/bin/env bash
PROJECT_ROOT=.
num_cpus=30
subword_bpe_merge_ops=600

# preprocess fine-tune train data
fairseq-preprocess \
        --source-lang en --target-lang de \
        --srcdict ${PROJECT_ROOT}/experiments/toy/vocab/vocab.bpe.${subword_bpe_merge_ops} \
        --tgtdict ${PROJECT_ROOT}/experiments/toy/vocab/vocab.bpe.${subword_bpe_merge_ops} \
        --trainpref ${PROJECT_ROOT}/experiments/toy/merged_data/en2de/train \
        --validpref ${PROJECT_ROOT}/experiments/toy/test_data/en2de/dev \
        --destdir ${PROJECT_ROOT}/experiments/toy/data/fine-tune/en2de \
        --workers ${num_cpus}

# preprocess fine-tune test data
fairseq-preprocess \
        --source-lang en --target-lang de \
        --srcdict ${PROJECT_ROOT}/experiments/toy/vocab/vocab.bpe.${subword_bpe_merge_ops} \
        --tgtdict ${PROJECT_ROOT}/experiments/toy/vocab/vocab.bpe.${subword_bpe_merge_ops} \
        --testpref ${PROJECT_ROOT}/experiments/toy/test_data/en2de/dev \
        --destdir ${PROJECT_ROOT}/experiments/toy/data/test/en2de/wmt14_head100/bin \
        --workers ${num_cpus}

mkdir -p ${PROJECT_ROOT}/experiments/toy/data/test/en2de/wmt14_head100/raw

cp ${PROJECT_ROOT}/experiments/toy/data/raw/test/en2de/* ${PROJECT_ROOT}/experiments/toy/data/test/en2de/wmt14_head100/raw/
