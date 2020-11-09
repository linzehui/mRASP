# Pre-Train and Fine-Tune
## Pre-requisite

Before this step, please ensure that you have finished the data preprocessing step.

```bash
bash ${PROJECT_ROOT}/train/pre-train.sh ${pretrain_config_yml}
```

```bash
bash ${PROJECT_ROOT}/train/fine-tune.sh ${finetune_config_yml} ${eval_config_yml}
```

## Pre-train & Fine-tune
### Example
#### pre-training config
```text
model_arch: transformer_wmt_en_de_big
encoder_learned_pos: true
decoder_learned_pos: true
data_path: /data00/home/panxiao.94/experiments/pmnmt/experiments/toy/data/pre-train
model_dir: /data00/home/panxiao.94/experiments/pmnmt/experiments/toy/models/pre-train/transformer_big
update_freq: 1
log_interval: 5
save_interval_updates: 50
max_tokens: 4096
max_source_positions: 256
max_target_positions: 256
lr: 5e-4
dropout: 0.2
activation_fn: gelu
criterion: label_smoothed_cross_entropy
reset_optimizer: true
reset_lr_scheduler: true
reset_dataloader: true
reset_meters: true
lr_scheduler: inverse_sqrt
weight_decay: 0.0
clip_norm: 0.0
warmup_init_lr: 1e-07
label_smoothing: 0.1
fp16: true
seed: 1
```
* `data_path` denotes the path of binarized training data.
* `model_dir` denotes the path where the checkpoints will be saved to.

#### fine-tuning config
```text
src: en
tgt: de
model_arch: transformer_wmt_en_de_big
encoder_learned_pos: true
decoder_learned_pos: true
data_path: /data00/home/panxiao.94/experiments/pmnmt/experiments/toy/data/fine-tune/en2de
model_dir: /data00/home/panxiao.94/experiments/pmnmt/experiments/toy/models/fine-tune/transformer_big/en2de
pretrain_model_dir: /data00/home/panxiao.94/experiments/pmnmt/experiments/toy/models/pre-train/transformer_big
update_freq: 1
log_interval: 5
save_interval_updates: 50
max_update: 500
max_tokens: 2048
max_source_positions: 256
max_target_positions: 256
lr: 5e-4
dropout: 0.2
activation_fn: gelu
criterion: label_smoothed_cross_entropy
reset_optimizer: true
reset_lr_scheduler: true
reset_dataloader: true
reset_meters: true
lr_scheduler: inverse_sqrt
weight_decay: 0.0
clip_norm: 0.0
warmup_init_lr: 1e-07
label_smoothing: 0.1
fp16: true
seed: 9818962
```
* `data_path` denotes the path of binarized training data.
* `model_dir` denotes the path where the checkpoints will be saved to.
* `pretrain_model_dir` denotes that path of the pre-trained model, please make sure that checkpoint_best.pt exists in this folder.
* `pretrain_model_ckpt` denotes the checkpoint that the fine-tune model will be initialized from. If both `pretrain_model_ckpt` and `pretrain_model_dir` are provided, `pretrain_model_ckpt` is used.

#### evaluation config
* used for evaluating
```text
beam_size: 5
eval_batch_size: 16
max_len_a: 0
max_len_b: 256
length_penalty: 0.5
test_data_path: /data00/home/panxiao.94/experiments/pmnmt/experiments/toy/data/test
testset: en2de/wmt14_head100
klist: 2:3:5
bleu_type: tok
post_command: "| sed -e s'|LANG_TOK_DE ||g' "
ref_tokenizer: MosesTokenizer
```
* `test_data_path` denotes the main path of test data
* `testset` denotes the sub-directory of the testset, the final testset path is `final_test_path=${test_data_path}/${testset}`, two directories should be placed under `${final_test_path}`: `raw` and `bin`.
    * `raw` denotes the initial raw data,  `dev.${tgt}` is used as a reference
    * `bin` denotes the binarized data that is ready for fairseq to use.
* `klist` denotes the values of k of top-k checkpoint average method, the values are separated by ':'
* `bleu_type` indicates the bleu calculation criterion: tok-bleu(`tok`) or detok-bleu(`detok`)

## Concatenate vocabulary
If you want to support a new language, you must add new tokens to the existing vocabulary and expand the embedding parameters in the checkpoint.
```bash
python ${PROJECT_ROOT}/train/scripts/concat_merge_vocab.py --checkpoint ${CKPT} --now-vocab ${CURRENT_VOCAB} --to-append-vocab ${NEW_VOCAB} --output-dir ${OUTPUT_DIR}
```
After running the above command, you will get a new checkpoint and a new vocab.
