#!/usr/bin/env bash

set -e

# repo_dir: root directory of the project
repo_dir="$( cd "$( dirname "$0" )" && pwd )"
echo "==== Working directory: ====" >&2
echo "${repo_dir}"
echo "============================" >&2
cd ${repo_dir}

# load training configs
training_config=$1
eval_config=$2
source ${repo_dir}/misc/load_config.sh ${training_config}
source ${repo_dir}/misc/load_config.sh ${eval_config}

IFS=',' read -r -a gpus <<< ${CUDA_VISIBLE_DEVICES}
N=${#gpus}

if [[ ! -d ${data_path} ]]; then
    echo "${data_path} not found" && exit 1;
fi

res_path=${model_dir}/results
log_file=${model_dir}/info.log
summary_file=${model_dir}/summary.log
mkdir -p ${res_path}

# reset
if [[ ${reset} == "true" ]]; then
    reset_cmd="--reset-optimizer"
else
    reset_cmd=""
fi

# lr
[[ -z ${warmup_updates} ]] && warmup_updates=4000


if [[ ! -z ${max_lr_start} && ${max_lr_start} == "true" ]] ; then
    max_lr_start_cmd="--max-lr-start"
else
    max_lr_start_cmd=""
fi

[[ -z ${min_lr} ]] && min_lr=1e-09

# dropout
dropout_cmd=""
for var in dropout input_dropout weight_dropout attention_dropout activation_dropout relu_dropout
do
    if [[ ! -z ${!var} ]]; then
        varname=`echo ${var} | sed 's/\_/\-/g'`
        dropout_cmd=${dropout_cmd}" --${varname}  ${!var}"
    fi
done

# restore from pretrain model
if [[ ! -z ${pretrain_model_ckpt} && -f ${pretrain_model_ckpt} ]]; then
    restore_ckpt=${pretrain_model_ckpt}
elif [[ ! -z ${pretrain_model_dir} && -d ${pretrain_model_dir} && -f ${pretrain_model_dir}/checkpoint_best.pt ]]; then
    restore_ckpt=${pretrain_model_dir}/checkpoint_best.pt
else
    echo "${pretrain_model_dir}/checkpoint_best.pt not found or ${pretrain_model_ckpt} not found" && exit 1;
fi

#
[[ -z ${seed} ]] && seed=1;
[[ -z ${label_smoothing} ]] && label_smoothing=0.1
[[ -z ${update_freq} ]] && update_freq=1
[[ -z ${log_interval} ]] && log_interval=100
[[ -z ${save_interval_updates} ]] && save_interval_updates=1000
[[ -z ${max_tokens} ]] && max_tokens=4096
[[ -z ${max_update} ]] && max_update=100000
[[ -z ${num_workers} ]] && num_workers=1


[[ ${encoder_learned_pos} == "true" ]] && encoder_learned_pos_option="--encoder-learned-pos"
[[ ${decoder_learned_pos} == "true" ]] && decoder_learned_pos_option="--decoder-learned-pos"
[[ ${fp16} == "true" ]] && fp16_option="--fp16"
[[ ${reset_optimizer} == "true" ]] && reset_optimizer_option="--reset-optimizer"
[[ ${reset_lr_scheduler} == "true" ]] && reset_lr_scheduler_option="--reset-lr-scheduler"
[[ ${reset_dataloader} == "true" ]] && reset_dataloader_option="--reset-dataloader"
[[ ${reset_meters} == "true" ]] && reset_meters_option="--reset-meters"

source ${repo_dir}/scripts/common_scripts.sh
source ${repo_dir}/misc/monitor.sh

command="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} fairseq-train ${data_path} \
  --seed ${seed} \
  -s ${src} \
  -t ${tgt} \
  --arch ${model_arch} \
  --share-all-embeddings \
  --restore-file ${restore_ckpt}
  ${encoder_learned_pos_option} \
  ${decoder_learned_pos_option} \
  --save-dir ${model_dir} \
  ${fp16_option} \
  --max-source-positions ${max_source_positions} \
  --max-target-positions ${max_target_positions} \
  ${dropout_cmd}  \
  --activation-fn ${activation_fn} \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm ${clip_norm} \
  --lr ${lr} --min-lr ${min_lr} --lr-scheduler ${lr_scheduler} --warmup-init-lr ${warmup_init_lr} --warmup-updates ${warmup_updates} \
  --criterion ${criterion} --label-smoothing ${label_smoothing} --weight-decay ${weight_decay} \
  ${max_lr_start_cmd} \
  --max-tokens ${max_tokens} \
  --update-freq ${update_freq} \
  --no-progress-bar --log-interval ${log_interval} \
  --save-interval-updates ${save_interval_updates} \
  --max-update ${max_update} \
  --skip-invalid-size-inputs-valid-test \
  ${reset_optimizer_option} \
  ${reset_lr_scheduler_option} \
  ${reset_dataloader_option} \
  ${reset_meters_option} \
  --num-workers ${num_workers} \
  --distributed-no-spawn \
  --ddp-backend no_c10d 1>&2
"
echo $command  >&2
eval $command

if [[ $? == 0 ]]; then
    echo "Fine-Tuning successfully finished" >&2
    source ${repo_dir}/scripts/average_ckpt.sh && echo "Average checkpoint successfully finished"
else
    echo "Some error happened when training, Retry It." >&2
    exit 1
fi
