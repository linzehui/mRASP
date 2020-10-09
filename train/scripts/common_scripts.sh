#!/usr/bin/env bash

get_tokenizer_cmd () {
    tokenizer=$1
    language=$2
    if [[ ${tokenizer} == "MosesTokenizer" ]]; then
        echo "| sacremoses tokenize -l ${language}"
    else
        echo "${tokenizer} not supported" && exit 1;
    fi
}

export -f get_tokenizer_cmd

get_detokenizer_cmd () {
    tokenizer=$1
    language=$2
    if [[ ${tokenizer} == "MosesTokenizer" ]]; then
        echo "| sacremoses detokenize -l ${language}"
    else
        echo "${tokenizer} not supported" && exit 1;
    fi
}

export -f get_detokenizer_cmd


bleu () {
    src=$1
    tgt=$2
    res_file=$3
    ref_file=$4
    type=$5
    bleu_type=$6
    n_process=$7

    if [[ -f ${res_file} ]]; then
        f_dirname=`dirname ${res_file}`
        python ${repo_dir}/scripts/rerank_utils.py ${res_file} ${ref_file} ${type} || exit 1;
        input_file="${f_dirname}/hypo.out.nobpe"
        if [[ ${bleu_type} == "detok" ]]; then
            detokenizer_cmd=`get_detokenizer_cmd ${ref_tokenizer} ${tgt}`
        elif [[ ${bleu_type} == "tok" ]]; then
            tokenizer_cmd=`get_tokenizer_cmd ${ref_tokenizer} ${tgt}`
        fi
        ref_cmd="cat ${f_dirname}/ref.out ${tokenizer_cmd}"
        eval ${ref_cmd} > "${f_dirname}/ref.out.final"
        hyp_cmd="cat ${input_file} ${post_command} ${detokenizer_cmd}"
        eval ${hyp_cmd} > "${f_dirname}/hypo.out.nobpe.final"
        cat "${f_dirname}/hypo.out.nobpe.final" | sacrebleu -l ${src}-${tgt} --short "${f_dirname}/ref.out.final" | awk '{print $3}'
    else
        echo "${res_file} not exist!" >&2 && exit 1;
    fi
}

export -f bleu

infer_test () {
    test_path=$1
    ckpts=$2
    gpu=$3
    final_res_file=$4
    if [[ $gpu -lt 0 ]]; then
        gpu_cmd="CUDA_VISIBLE_DEVICES= "
        cpu_cmd="--cpu"
    else
        gpu_cmd="CUDA_VISIBLE_DEVICES=${gpu} "
    fi
    command=${gpu_cmd}"fairseq-generate ${test_path} \
    -s ${src} \
    -t ${tgt} \
    --skip-invalid-size-inputs-valid-test \
    --path ${ckpts} \
    --batch-size ${eval_batch_size} \
    --beam ${beam_size} \
    --nbest 1 \
    ${cpu_cmd} \
    --lenpen ${length_penalty} \
    --max-len-a ${max_len_a} \
    --max-len-b ${max_len_b} \
    --max-source-positions ${max_source_positions} \
    --max-target-positions ${max_target_positions} | grep -E '[S|H|P|T]-[0-9]+' > ${final_res_file}
    "
    echo "$command"
}

export -f infer_test

