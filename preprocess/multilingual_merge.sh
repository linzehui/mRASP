#!/usr/bin/env bash

[[ ! -d ${merged_output_path} ]] && mkdir -p ${merged_output_path}

echo "===== TYPE: ManyToMany ====="
IFS=';' read -r -a direcs <<< ${directions}
IFS=';' read -r -a langs <<< ${languages}
declare -a source_files=()
declare -a new_source_files=()
declare -a new_target_files=()

echo "-->=== Add Lang Token to front of Training Corpus ===<--"
for direc in "${direcs[@]}"
do
    IFS='2' read -r -a langs <<< ${direc}
    src_lang_token="LANG_TOK_"`echo "${langs[0]}" | tr '[a-z]' '[A-Z]'`
    trg_lang_token="LANG_TOK_"`echo "${langs[1]}" | tr '[a-z]' '[A-Z]'`
    if [[ ${file_prefix} == "train" ]]; then
        subdir="${langs[0]}_${langs[1]}"
        _subdir="${langs[1]}_${langs[0]}"
        [[ ! -d ${output_path}/${subdir} ]] && subdir=${_subdir}
    else
        subdir=${direc}
    fi
    from_path=${output_path}/${subdir}

    src_file="${from_path}/${file_prefix}.${langs[0]}"
    trg_file="${from_path}/${file_prefix}.${langs[1]}"
    base_src_filename=`basename ${src_file}`
    new_src_filename="${from_path}/new_${base_src_filename}"
    if [[ ! -f ${new_src_filename} ]]; then
        echo "${base_src_filename} --> `basename ${new_src_filename}`"
        sed -e 's/^/'${src_lang_token}' /' ${src_file} > ${new_src_filename}
    fi
    new_source_files+="${new_src_filename} "

    ## trg
    base_trg_filename=`basename ${trg_file}`
    new_trg_filename="${from_path}/new_${base_trg_filename}"
    if [[ ! -f ${new_trg_filename} ]]; then
        echo "${base_trg_filename} --> `basename ${new_trg_filename}`"
        # add trg lang token
        sed -e 's/^/'${trg_lang_token}' /' ${trg_file} > ${new_trg_filename}
    fi
    # if subword is not applied on target file
    new_target_files+="${new_trg_filename} "
done

echo "==== Concatenating to final path ===="
cat ${new_source_files} > ${merged_output_path}/${file_prefix}.src
cat ${new_target_files} > ${merged_output_path}/${file_prefix}.trg


if [[ ${file_prefix} == "train" && ${learn} == "true" ]]; then
    echo "-->=== Add Lang Token to Vocab ===<--"
    # add LANG TOKEN to vocab
    final_vocabfile=${final_vocab_path}/vocab.bpe.${subword_bpe_merge_ops}
    IFS=';' read -r -a langs <<< ${languages}
    for lang in "${langs[@]}"
    do
        echo -e "LANG_TOK_"`echo "${lang}" | tr '[a-z]' '[A-Z]'`"\t-1" >> ${final_vocabfile}
    done
fi


echo "" >&2
echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== BEGIN shuffling Merged Dataset =====" >&2
random_source=`date +%N`
shuf --random-source=<(get_seeded_random $random_source) ${merged_output_path}/${file_prefix}.src > ${merged_output_path}/${file_prefix}.src.shuf
mv ${merged_output_path}/${file_prefix}.src.shuf ${merged_output_path}/${file_prefix}.src
shuf --random-source=<(get_seeded_random $random_source) ${merged_output_path}/${file_prefix}.trg > ${merged_output_path}/${file_prefix}.trg.shuf
mv ${merged_output_path}/${file_prefix}.trg.shuf ${merged_output_path}/${file_prefix}.trg
echo "===== $(date "+%Y-%m-%d %H:%M:%S") ===== FINISH shuffling Merged Dataset =====" >&2

rm ${new_source_files}

for _f in ${new_target_files}
do
    echo ${_f}
    [[ -f ${_f} ]] && rm ${_f}
done
