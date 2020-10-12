#!/usr/bin/env bash

set -e

# repo_dir: <PROJECT_ROOT>/preprocess
config_yml=$1
# language: en
# tokenizer: MosesTokenizer
# do_normalize_punctuations: true
#

input_file=$2
output_path=$3

# read variables from yaml file
eval $(parse_yaml ${config_yml})

# set scripts path
tokenize_script_dir=${repo_dir}/tools/data_preprocess/tokenize_scripts
clean_script_dir=${repo_dir}/tools/data_preprocess/clean_scripts

process_cmd="cat ${input_file}"

echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== Start Clean and Tokenization for language=${language} ====="
source ${repo_dir}/tools/data_preprocess/clean_each.sh
source ${repo_dir}/tools/data_preprocess/tokenize_each.sh

# run process command
file_out=${output_path}/${file_prefix}.${language}
process_cmd="${process_cmd} > ${file_out}"
echo "Preprocess Command: ${process_cmd}"
echo "    applying above... to ${file_out}"
eval ${process_cmd}

echo "    = $(date "+%Y-%m-%d %H:%M:%S") === Clean and Tokenization FINISHED ==="