#!/usr/bin/env bash

input_path=$1
output_path=$2

[[ ! -d ${output_path} ]] && mkdir -p ${output_path}
codes_file=${final_vocab_path}/codes.bpe.${subword_bpe_merge_ops}

IFS=';' read -r -a lang_pairs <<< ${pairs}

echo "INPUT DIR: ${input_path}"
echo "OUTPUT DIR: ${output_path}"
#for pair in "${lang_pairs[@]}"
for subdir in `ls -d ${input_path}/*`
do
    subdir=`basename ${subdir}`
    if [[ $subdir =~ [a-z]+_[a-z]+ || $subdir =~ [a-z]+2[a-z]+ ]]; then
        input_dir=${input_path}/${subdir}
        # create sub directory if not exists
        [[ ! -d ${output_path}/${subdir} ]] && mkdir -p ${output_path}/${subdir} && echo "created ${output_path}/${subdir}"
        file_list="${input_dir}/*"
        for file in `ls ${file_list}`
        do
            base_filename=`basename $file`
            output_file="${output_path}/${subdir}/${base_filename}"
            # apply subword
            ## input: the original file to be encoded; output: the encoded file
            cat ${file} | mulitprocess_pipeline "subword-nmt apply-bpe -c ${codes_file}" ${num_cpus} > ${output_file}
        done
    fi
    echo "SUBDIR: ${subdir} done"
done
