#!/usr/bin/env bash

dict_path_root=`dirname ${ras_dict_path}`
dict_path_basename=`basename ${ras_dict_path}`

# get MUSE dictionaries
echo "Down load MUSE dictionaries ... ..."
cd ${dict_path_root}
if [[ ! -d ${dict_path_basename} ]]; then
    if [[ ! -f dictionaries.tar.gz ]]; then
        wget https://dl.fbaipublicfiles.com/arrival/dictionaries.tar.gz
    fi
    mkdir ${dict_path_basename} ${dict_path_basename}_tmp
    tar -xvzf dictionaries.tar.gz -C ${dict_path_basename}_tmp
    mv ${dict_path_basename}_tmp/dictionaries/[a-z]?-[a-z]?.txt ${dict_path_basename}/
    rm -r ${dict_path_basename}_tmp
    if [[ -f dictionaries.tar.gz ]]; then
        rm dictionaries.tar.gz
    fi
fi

cd ${repo_dir}

command=""

for varname in dict_path replace_prob num_repeat vocab_size
do
    ras_varname=ras_${varname}
    if [[ ${!ras_varname} ]]; then
     command=${command}" --${varname} ${!ras_varname}"
    fi
done

codes_file=${final_vocab_path}/codes.bpe.${subword_bpe_merge_ops}
python -m tools.ras.replace_word --langs ${languages} ${command} --data_path ${merged_output_path}
cat ${merged_output_path}/expanded_train.src | mulitprocess_pipeline "subword-nmt apply-bpe -c ${codes_file}" ${num_cpus} > ${merged_output_path}/expanded_train_bpe.src
paste -d ' ' ${merged_output_path}/lang_indicator.src ${merged_output_path}/expanded_train_bpe.src > ${merged_output_path}/expanded_train_final.src
rm ${merged_output_path}/lang_indicator.src ${merged_output_path}/expanded_train_bpe.src ${merged_output_path}/expanded_train.src
mv ${merged_output_path}/expanded_train_final.src ${merged_output_path}/expanded_train.src
cat ${merged_output_path}/expanded_train.src >> ${merged_output_path}/train.src
cat ${merged_output_path}/expanded_train.trg >> ${merged_output_path}/train.trg
rm ${merged_output_path}/expanded_train.*

