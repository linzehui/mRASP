#!/usr/bin/env bash

set -e

# repo_dir: root directory of the project
repo_dir="$( cd "$( dirname "$0" )" && pwd )"
echo "==== Working directory: ====" >&2
echo "${repo_dir}"
echo "============================" >&2
cd ${repo_dir}

# get common utils
source ${repo_dir}/tools/common.sh

main_config_yml=$1
# raw_data_path
# output_main_path
# merged_output_path
# output_sub_path: prep
# configs_subdir: configs
# cleaned_subdir: cleaned
# final_vocab_path
# logs_subdir: logs
# vocab_subdir: vocab
# output_subdir: output
# preprocess_steps_list: clean:subword:merge:ras
# file_prefix
# learn
# subword_bpe_merge_ops
# maximum_vocabulary_size
# minimum_frequency
# pairs
# directions
# languages

eval $(parse_yaml ${main_config_yml})

# setup paths
configs_path=${output_main_path}/${configs_subdir}
cleaned_path=${output_main_path}/${cleaned_subdir}
logs_path=${output_main_path}/${logs_subdir}
output_path=${output_main_path}/${output_subdir}
vocab_path=${output_main_path}/${vocab_subdir}

IFS=':' read -r -a preprocess_steps <<< ${preprocess_steps_list}

prefix="preprocess"

# check existence of data_path and configs_path
[[ ! -d ${raw_data_path} ]] && echo "Data Path ${raw_data_path} does not exist!" && exit 1
[[ ! -d ${configs_path} ]] && mkdir -p ${configs_path}

# create log path if not exists
[[ ! -d ${logs_path} ]] && mkdir -p ${logs_path}

# read in all language pairs
IFS=';' read -r -a lang_pairs <<< ${pairs}
IFS=';' read -r -a direcs <<< ${directions}
IFS=';' read -r -a langs <<< ${languages}

if [[ " ${preprocess_steps[@]} " =~ "clean"  ]]; then
    echo "======== 1. Clean & Tokenization BEGIN ========"
    # put clean_data.sh to root path
    # 1. clean each parallel corpus
    # 1.1 generate preprocess config yaml file for each language and clean monolingual data cleaning
    echo "******** Generate config for LANGUAGE ${lang} ********"
    for lang in "${langs[@]}";
    do
        python ${repo_dir}/tools/misc/multilingual_preprocess_yml_generator.py ${main_config_yml} ${lang} ${file_prefix}
    done

    if [[ ${learn} == "true" || ${mono} == "true" ]]; then
        for lang in "${langs[@]}"
        do
            echo "******** Clean & Tokenize ${lang} Mono Data ********"
            if [[ ${file_prefix} == "train" ]]; then
                (python ${repo_dir}/tools/misc/multilingual_preprocess_yml_generator.py ${main_config_yml} ${lang} ${file_prefix} &&
                source ${repo_dir}/tools/data_preprocess/prep_mono.sh ${configs_path}/preprocess_${lang}.yml ${raw_data_path}/${lang} ${cleaned_path}/${lang} &>${logs_path}/1_preprocess_mono_${lang}.log)
            else
                python ${repo_dir}/tools/misc/multilingual_preprocess_yml_generator.py ${main_config_yml} ${lang} ${file_prefix}
            fi
        done

        wait
    fi

    # 1.2. generate preprocess config yaml file and preprocess parallel data in parallel for each language pair

    if [[ ${file_prefix} == "train" ]]; then
        subdirs=("${lang_pairs[@]}") # training data
    else
        subdirs=("${direcs[@]}") # dev data
    fi


    for subdir in "${subdirs[@]}"
    do
        echo "******** Generate config for LANGUAGE PAIR ${subdir} and Clean & Tokenize ${subdir} Parallel Data ********"
        (python ${repo_dir}/tools/misc/multilingual_preprocess_yml_generator.py ${main_config_yml} ${subdir} ${file_prefix} &&
        source ${repo_dir}/tools/data_preprocess/prep_parallel.sh ${configs_path}/preprocess_${subdir}.yml ${cleaned_path}/${subdir} &>${logs_path}/1_preprocess_parallel_${subdir}.log) &
    done
    wait
    echo "======== 1. Clean & Tokenization ALL DONE ========"
fi


# 2. jointly learn subword and vocab, and apply

if [[ " ${preprocess_steps[@]} " =~ "subword"  ]]; then
    echo "======== 2. Subword & Vocab BEGIN ========"
    if [[ ${learn} == "true" ]]; then
        # todo: balance data before learning vocab
        echo "******** Learn + Apply BEGIN ********"
        source ${repo_dir}/tools/subword/multilingual_learn_apply_subword_vocab_joint.sh ${cleaned_path} ${output_path} &>${logs_path}/2_preprocess_learn_apply.log
        echo "******** Learn + Apply ALL DONE ********"
        echo "******** Copy Vocab to Final Path BEGIN ********"
        [[ ! -d ${final_vocab_path} ]] && mkdir -p ${final_vocab_path}
        cp  ${vocab_path}/* ${final_vocab_path}
        echo "******** Copy Vocab to Final Path ALL DONE ********"
    else
        echo "******** Only Apply BEGIN ********"
        source ${repo_dir}/tools/subword/multilingual_apply_subword_vocab.sh ${cleaned_path} ${output_path} &>${logs_path}/2_preprocess_apply.log
        echo "******** Only Apply ALL DONE ********"
    fi
    sed -ri s"|\t| |g" ${final_vocab_path}/vocab.bpe.${subword_bpe_merge_ops}
    echo "======== 2. Subword & Vocab ALL DONE ========"
fi

# 3. special preprocessing operations for multilingual NMT (add token indicating the target language)
if [[ " ${preprocess_steps[@]} " =~ "merge"  ]]; then
    echo "======== 3. Merge BEGIN ========"
    source ${repo_dir}/multilingual_merge.sh &>${logs_path}/3_preprocess_merge.log
    echo "======== 3. Merge ALL DONE ========"
fi

# 4. RAS
if [[ ${file_prefix} == "train" && " ${preprocess_steps[@]} " =~ "ras" ]]; then
    if [[ -z ${ras_multi_dict_path} ]]; then
        echo "======== 4. Random Alignment Substitution BEGIN ========"
        source ${repo_dir}/tools/ras/random_alignment_substitution.sh &>${logs_path}/4_ras.log
        echo "======== 4. Random Alignment Substitution ALL DONE ========"
    else
        echo "======== 4. Random Alignment Substitution W/ Multi BEGIN ========"
        source ${repo_dir}/tools/ras/random_alignment_substitution_w_multi.sh &>${logs_path}/4_ras_multi.log
        echo "======== 4. Random Alignment Substitution W/ Multi ALL DONE ========"
    fi
fi

rm -r ${repo_dir}/mp_tmp

if [[ ! -z ${src} && ! -z ${trg} ]]; then
    mv ${merged_output_path}/${file_prefix}.src ${merged_output_path}/${file_prefix}.${src}
    mv ${merged_output_path}/${file_prefix}.trg ${merged_output_path}/${file_prefix}.${trg}
fi
