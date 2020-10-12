#!/usr/bin/env bash

set -e

config_yml=$1
# shuffle
# deduplicate
# keep_lines_percent
#
# language1:
#   language
#   file
#   config_file
# language2:
#   language
#   file
#   config_file

eval $(parse_yaml ${config_yml})

output_path=$2
script_dir=${repo_dir}/tools/data_preprocess/clean_scripts
output_tmp=${output_path}/tmp

echo "== output tmp dir:  ${output_tmp} ==" >&2
echo "== output dir:  ${output_path} ==" >&2

[[ ! -d ${output_tmp} ]] && mkdir -p ${output_tmp};
[[ ! -d ${output_path} ]] && mkdir -p ${output_path};

echo "Preprocessing the dataset..." >&2


# deduplicate
if [[ ${deduplicate} == "true" ]]; then
    echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== Process option deduplicate=true =====" >&2
    echo "  Call paste to merge" >&2
    echo "    ${language1_file}," >&2
    echo "    ${language2_file}" >&2
    echo "    to ${output_tmp}/merged" >&2
    paste -d "\t" ${language1_file} ${language2_file} > ${output_tmp}/merged
    echo "  Call sort -u to sort and deduplicate file" >&2
    echo "    to ${output_tmp}/merged.sort" >&2
    sort -u ${output_tmp}/merged > ${output_tmp}/merged.sort
    rm ${output_tmp}/merged
    language1_file=${output_tmp}/${language1_file##*/}.uniq
    language2_file=${output_tmp}/${language2_file##*/}.uniq
    echo "  Call cut -f 1/2 to split file" >&2
    echo "    to ${language1_file}," >&2
    echo "    ${language2_file}" >&2
    cut -f 1 ${output_tmp}/merged.sort > ${language1_file}
    cut -f 2 ${output_tmp}/merged.sort > ${language2_file}
    rm ${output_tmp}/merged.sort
    echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== FINISH option deduplicate=true =====" >&2
    to_remove="true"
fi

echo "===== Start Preprocess Data =====" >&2
source ${repo_dir}/tools/data_preprocess/prep_each.sh ${language1_config_file} ${language1_file} ${output_path} >&2 &
source ${repo_dir}/tools/data_preprocess/prep_each.sh ${language2_config_file} ${language2_file} ${output_path} >&2
wait

echo "===== End Preprocess Data =====" >&2


if [[ ${shuffle} == "true" ]]; then
    for file in $(ls ${output_path}); do
        if [[ ${file} =~ "${file_prefix}.${language1_language}" ]]; then
            clean_file1=${output_path}/${file}
        elif [[ ${file} =~ "${file_prefix}.${language2_language}" ]]; then
            clean_file2=${output_path}/${file}
        fi
    done
    echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== BEGIN shuffling =====" >&2
    random_source=`date +%N`
    shuf --random-source=<(get_seeded_random $random_source) ${clean_file1} > ${output_tmp}/${file_prefix}.${language1_language}.shuf
    mv ${output_tmp}/${file_prefix}.${language1_language}.shuf ${output_path}/${file_prefix}.${language1_language}
    shuf --random-source=<(get_seeded_random $random_source) ${clean_file2} > ${output_tmp}/${file_prefix}.${language2_language}.shuf
    mv ${output_tmp}/${file_prefix}.${language2_language}.shuf ${output_path}/${file_prefix}.${language2_language}
    echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== FINISH shuffling =====" >&2
fi

rm -r ${output_tmp}