#!/usr/bin/env bash

input_path=$1
output_path=$2

output_tmp=${output_path}/tmp

[[ ! -d ${output_tmp} ]] && mkdir -p ${output_tmp}

[[ ! -d ${output_path} ]] && mkdir -p ${output_path}

# get a list of all cleaned data
# declare -a all_files=()
# declare -a all_output_files=()
# declare -a extra_output_files=()  # monolingual corpus used to compute vocab

IFS=';' read -r -a lang_pairs <<< ${pairs}
IFS=';' read -r -a langs <<< ${languages}

# parallel data
for pair in "${lang_pairs[@]}"
do
    IFS='_' read -r -a lgs <<< ${pair}
    subdir="${lgs[0]}_${lgs[1]}"
    _subdir="${lgs[1]}_${lgs[0]}"
    input_dir=${input_path}/${subdir}
    [[ ! -d ${input_path}/${subdir} ]] && input_dir=${input_path}/${_subdir}
    echo "INPUT PATH: ${input_dir}"
    # create sub directory if not exists
    [[ ! -d ${output_path}/${subdir} ]] && mkdir -p ${output_path}/${subdir}

    for file in ${input_dir}/*
    do
        base_filename=`basename $file`
        all_files+="${file};"
        all_parallel_files+="${file} "
    done
done


# Add some monolingual data to learn subword and vocabulary together

for lang in "${langs[@]}"
do
    input_dir=${input_path}/${lang}
    echo "INPUT PATH: ${input_dir}"
    # create sub directory if not exists
    for file in ${input_dir}/*
    do
        base_filename=`basename $file`
        all_files+="${file};"
    done
done

declare -A lang_count
declare -A lang_files

## resample files to balanced volume
echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== Resample Corpus ====="
IFS=';' read -r -a fs <<< ${all_files}
for file in "${fs[@]}"
do
    [[ -d $file ]] && continue;
    base_filename=`basename $file`
    lang="${base_filename##*.}"
    lang_files["$lang"]+="$file "
    let lang_count["$lang"]=`wc -l < $file`+lang_count["$lang"]
done

max_cnt=0
for lang in "${!lang_count[@]}"
do
    if [[ ${lang_count["${lang}"]} -gt ${max_cnt} ]]; then
        max_cnt=${lang_count["${lang}"]}
    fi
    echo "$lang: ${lang_count["${lang}"]}"
    tmp_file=${output_tmp}/all.${lang}
    cat ${lang_files["${lang}"]} > ${tmp_file}
done

echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== Max Count: ${max_cnt} ====="

#declare -a all_input_files=()
#declare -a all_tmp_output_files=()
for lang in "${!lang_count[@]}"
do
    tmp_file=${output_tmp}/all.${lang}
    final_file=${output_tmp}/final.${lang}
    tmp_output_file=${output_tmp}/out.${lang}
    truncate -s 0 ${final_file}  # clear the file
    file_cnt=${lang_count["${lang}"]}
    remain_cnt=${max_cnt}
    # up-sample
    while [[ ${remain_cnt} -ge ${file_cnt} ]]
    do
        cat ${tmp_file} >> ${final_file}
        remain_cnt=$[${remain_cnt}-${file_cnt}]
    done
    shuf -n ${remain_cnt} ${tmp_file} >> ${final_file}
    all_input_files+="${final_file} "
    all_tmp_output_files+="${tmp_output_file} "
done

echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== FINISH Resample ====="

[[ ! -d ${vocab_path} ]] && mkdir -p ${vocab_path}
codes_file=${vocab_path}/codes.bpe.${subword_bpe_merge_ops}

# jointly learn subword
echo "" >&2
echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== Learn Joint BPE and Apply =====" >&2
# we assume the bpe_merge_ops of both side are the same
python -m tools.subword.scripts.multilingual_learn_joint_bpe_and_vocab -i ${all_input_files} \
        --write-subword-file ${all_tmp_output_files} \
        --write-vocabulary ${vocab_path}/vocab.bpe.${subword_bpe_merge_ops}.init \
        -s ${subword_bpe_merge_ops} \
        -o ${codes_file} \
        --threads ${num_cpus}
echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== FINISH Learn Joint BPE and Apply =====" >&2

echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== Apply BPE to original input file =====" >&2
for file in ${all_parallel_files}
do
    base_filename=`basename $file`
    dirname=`dirname $file`
    subdir=`basename $dirname`
    echo "${subdir}   ${base_filename}"
    output_file="${output_path}/${subdir}/${base_filename}"
    # apply subword
    ## input: the original file to be encoded
    ## output: the encoded file
    cat ${file} | mulitprocess_pipeline "subword-nmt apply-bpe -c ${codes_file}" ${num_cpus} > ${output_file}
done
echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== FINISH Apply BPE to original input file =====" >&2

# Remove infrequent tokens from vocabulary
echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== Cut Vocabulary =====" >&2
python -m tools.subword.scripts.cut_vocab --max_vocab_size ${maximum_vocabulary_size} --min_frequency ${minimum_frequency} \
        -i ${vocab_path}/vocab.bpe.${subword_bpe_merge_ops}.init -o ${vocab_path}/vocab.bpe.${subword_bpe_merge_ops}
rm ${vocab_path}/vocab.bpe.${subword_bpe_merge_ops}.init
echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== FINISH Cut Vocabulary =====" >&2

# Clear up temporary files
rm ${all_input_files}
rm ${all_tmp_output_files}
rm -r ${output_tmp}