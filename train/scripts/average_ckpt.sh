#!/usr/bin/env bash


# 3. generate averaged ckpt
## read summary.log file
avg_model_list=""
IFS=':' read -r -a k_list <<< ${klist}
for k in "${k_list[@]}"
do
    if [[ -f ${model_dir}/summary.log ]]; then
        cmd="cat ${model_dir}/summary.log  2> /dev/null"
    else
        echo "Please ensures that ${model_dir}/summary.log exists!" >&2 && exit 1;
    fi

    ckpts_input=""
    for ckpt in `eval "$cmd | python ${repo_dir}/scripts/keep_top_ckpt.py '${testset}' ${k} False True"`
    do
        ckpt_path=${model_dir}/${ckpt}.pt
        ckpts_input="${ckpts_input} ${ckpt_path}"
    done

    # set model name
    idx=0

    model_out_name=checkpoint_avg_${k}best
    ckptname=${model_out_name}_${idx}

    while [[ -f ${model_dir}/${model_out_name}_${idx}.pt ]]
    do
        idx=$((idx+1));
    done

    ckptname=${model_out_name}_${idx}

    ## average checkpoint
    python ${repo_dir}/scripts/average_checkpoints_from_file.py --inputs ${ckpts_input} --output ${model_dir}/${ckptname}.pt
    avg_model_list=${avg_model_list}:${model_dir}/${ckptname}.pt  # add to model list
done


IFS=':' read -r -a avg_modellist <<< ${avg_model_list}

(for ckpt in ${avg_modellist[@]}
do
    ((i=i%N)); ((i++==0)) && wait
    ckptbase=`basename $ckpt`
    ckptname="${ckptbase%.*}"
    test_path=${test_data_path}/${testset}/bin
    testname=`echo ${testset} | sed 's/\//\_\_/g'`
    echo "-----> "$ckptname" | "$testname" <-----" >&2
    if [[ ! -d ${res_path}/${ckptname} ]]; then
        mkdir -p ${res_path}/${ckptname}
    fi
    final_res_file="${res_path}/${ckptname}/${testname}.txt"
    command=`infer_test ${test_path} ${model_dir}/${ckptname}.pt $((i-1)) ${final_res_file}`
    eval $command &
done)
