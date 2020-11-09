#!/usr/bin/env bash

cd ${repo_dir}

command=""

for varname in multi_dict_path replace_prob num_repeat vocab_size max_dep
do
    ras_varname=ras_${varname}
    if [[ ${!ras_varname} ]]; then
     command=${command}" --${varname} ${!ras_varname}"
    fi
done

# use default if `ras_languages` and `ras_target_languages` is not set
[[ -z ${ras_languages} ]] && langs=${languages} || langs=${ras_languages}
[[ -z ${ras_target_languages} ]] && target_langs=${languages} || target_langs=${ras_target_languages}

codes_file=${final_vocab_path}/codes.bpe.${subword_bpe_merge_ops}
echo "--langs ${languages} --target-langs ${languages} ${command} --data_path ${merged_output_path}"
python -m tools.ras.replace_word_w_multi --langs ${langs} --target-langs ${target_langs} ${command} --data_path ${merged_output_path}
cat ${merged_output_path}/expanded_train.src | mulitprocess_pipeline "subword-nmt apply-bpe -c ${codes_file}" ${num_cpus} > ${merged_output_path}/expanded_train_bpe.src
paste -d ' ' ${merged_output_path}/lang_indicator.src ${merged_output_path}/expanded_train_bpe.src > ${merged_output_path}/expanded_train_final.src
rm ${merged_output_path}/lang_indicator.src ${merged_output_path}/expanded_train_bpe.src ${merged_output_path}/expanded_train.src
mv ${merged_output_path}/expanded_train_final.src ${merged_output_path}/expanded_train.src
if [[ ${mono} == "true" ]]; then
    cat ${merged_output_path}/expanded_train.src > ${merged_output_path}/train.src
    cat ${merged_output_path}/expanded_train.trg > ${merged_output_path}/train.trg
else
    cat ${merged_output_path}/expanded_train.src >> ${merged_output_path}/train.src
    cat ${merged_output_path}/expanded_train.trg >> ${merged_output_path}/train.trg
fi
rm ${merged_output_path}/expanded_train.*

