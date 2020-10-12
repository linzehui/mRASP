#!/usr/bin/env bash

set -e

config_yml=$1
input_path=$2
output_path=$3

eval $(parse_yaml ${config_yml})


echo "== Output Path:  ${output_path} ==" >&2

[[ ! -d ${output_path} ]] && mkdir -p ${output_path}


echo "Preprocessing the dataset..." >&2

echo "===== Start Preprocess Data =====" >&2
for file in ${input_path}/*
do
    echo `basename ${file}`
    source ${repo_dir}/tools/data_preprocess/prep_each.sh ${config_yml} ${file} ${output_path} >&2 &
done
wait

echo "===== End Preprocess Data =====" >&2
