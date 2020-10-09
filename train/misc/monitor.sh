#!/usr/bin/env bash

[[ -z ${EVAL_GPU_INDEX} ]] && EVAL_GPU_INDEX=-1

(inotifywait -m ${model_dir} -e close_write |
while read path action file; do
    if [[ "$file" =~ .*pt$ ]]; then
        # eval with cpu
        ckptbase=${file}
        ckptname="${ckptbase%.*}"
        if [[ ${ckptname} != "checkpoint_last" && ${ckptname} != "checkpoint_best" && ${ckptname} =~ checkpoint_* ]]; then
            test_path=${test_data_path}/${testset}/bin
            testname=`echo ${testset} | sed 's/\//\_\_/g'`
            echo "-----> "${ckptname}" | "${testname}" <-----" >&2
            [[ ! -d ${res_path}/${ckptname} ]] && mkdir -p ${res_path}/${ckptname}
            final_res_file="${res_path}/${ckptname}/${testname}.txt"
            command=`infer_test ${test_path} ${model_dir}/${ckptname}.pt ${EVAL_GPU_INDEX} ${final_res_file}`
            eval ${command} || { echo "fairseq-generate FAILED !"; exit 1; }
        fi
    fi
done) &

echo "------ $(date "+%Y-%m-%d %H:%M:%S") ------" >> ${log_file}
(inotifywait -r -m -e close_write ${res_path} |
while read path action file; do
    if [[ "$file" =~ .*txt$ ]]; then
        ckptname=`basename $path`
        echo -e "$(date "+%Y-%m-%d %H:%M:%S")\t result file detected: $ckptname/$file" >> ${log_file}
        testname="${file%.*}"
        testname=`echo "$testname" | sed 's/\_\_/\//g'`
        res_file=${path}${file}
        ref_file=${test_data_path}/${testname}/raw/dev.${tgt}
        bleuscore=`bleu ${src} ${tgt} ${res_file} ${ref_file} l2r ${bleu_type} ${num_cpus}`
        bleu_str="$(date "+%Y-%m-%d %H:%M:%S")\t${ckptname}\t${testname}\t$bleuscore"
        echo -e ${bleu_str} >> ${summary_file}
        echo -e ${bleu_str}  # to stdout
    fi
done) &
