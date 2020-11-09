#!/usr/bin/env bash
PROJECT_ROOT=.
num_cpus=30
subword_bpe_merge_ops=600

# preprocess pre-train train data
fairseq-preprocess \
        --source-lang src --target-lang trg \
        --srcdict ${PROJECT_ROOT}/experiments/toy/vocab/vocab.bpe.${subword_bpe_merge_ops} \
        --tgtdict ${PROJECT_ROOT}/experiments/toy/vocab/vocab.bpe.${subword_bpe_merge_ops} \
        --trainpref ${PROJECT_ROOT}/experiments/toy/merged_data/train \
        --validpref ${PROJECT_ROOT}/experiments/toy/merged_data/dev \
        --destdir ${PROJECT_ROOT}/experiments/toy/data/pre-train \
        --workers ${num_cpus}
