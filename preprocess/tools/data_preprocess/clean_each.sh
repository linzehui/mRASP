#!/usr/bin/env bash


echo "  (by default)deescape_special_fields=true and remove-non-printing-chars=true"
process_cmd="${process_cmd} | perl ${clean_script_dir}/deescape-and-remove-nonprint.pl -threads ${num_cpus} "

if [[ ${do_normalize_punctuations} == "true" ]]; then
    echo "  do_normalize_punctuations=${do_normalize_punctuations}"
    process_cmd="${process_cmd} | perl ${clean_script_dir}/normalize-punctuation.pl -l ${language} -threads ${num_cpus}"
fi

process_cmd="${process_cmd} | sed 's/[ ][ ]\+/ /g'"

