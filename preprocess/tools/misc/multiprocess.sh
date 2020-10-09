#!/usr/bin/env bash

MULTIPROCESS_PIPELINE_DIR=${repo_dir}/mp_tmp
mkdir -p ${MULTIPROCESS_PIPELINE_DIR}

mulitprocess_pipeline(){
    cat > ${MULTIPROCESS_PIPELINE_DIR}/mp_infile
    echo "Now have read all input data" >&2

    cmd=$1
    threads=$2
    if [[ -z ${threads} ]]; then
        threads=${num_cpus}
    fi

    tmp_dir=${MULTIPROCESS_PIPELINE_DIR}/mp
    [[ -d ${tmp_dir} ]] && rm -r ${tmp_dir}
    mkdir -p ${tmp_dir}
    mkdir -p ${tmp_dir}/in
    mkdir -p ${tmp_dir}/out
    echo "split all data to different process" >&2
    split -n l/${threads} ${MULTIPROCESS_PIPELINE_DIR}/mp_infile ${tmp_dir}/in/in_file.

    for f in `ls ${tmp_dir}/in`; do
        name=`basename ${f}`
        echo "start the process ${name}" >&2
        cat ${tmp_dir}/in/${f} | eval "${cmd}" > ${tmp_dir}/out/${name} &
    done

    echo "Wait all process end" >&2
    wait
    echo "All process ended" >&2

    echo "Write the output to the standard output" >&2
    cat ${tmp_dir}/out/in_file*
}

export -f mulitprocess_pipeline
