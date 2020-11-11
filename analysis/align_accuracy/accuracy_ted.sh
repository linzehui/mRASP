#!/usr/bin/env bash
# repo_dir: root directory of the project
repo_dir="$( cd "$( dirname "$0" )" && pwd )"
echo "==== Working directory: ====" >&2
echo "${repo_dir}" >&2
echo "============================" >&2

# download ted 15-way parallel test set: 2284 sentences for each language.
# ar,cs,de,en,es,fr,it,ja,ko,nl,ro,ru,tr,vi,zh
# todo


# settings, modify to your location before running the script
MRASP_DATA_PATH="${repo_dir}/ted_multiway/bin/mrasp"
CKPT_YAML="${repo_dir}/model_ckpts.yml"
SAVEDIR="${repo_dir}/sent_vec"


# load checkpoints for mrasp-align, mrasp-noalign, and mbart
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

eval $(parse_yaml ${CKPT_YAML})


# mRASP align
for lang in ar cs de en es fr it ja ko nl ro ru tr vi zh
do
    python get_sent_vec.py ${MRASP_DATA_PATH} \
    --path ${align} --key align \
    --gen-subset test --task translation \
    -s ${lang} -t en \
    --batch-size 16 \
    --savedir ${SAVEDIR}
done

# mRASP align
for lang in ar cs de en es fr it ja ko nl ro ru tr vi zh
do
    python get_sent_vec.py ${MRASP_DATA_PATH} \
    --path ${noalign} --key noalign \
    --gen-subset test --task translation \
    -s ${lang} -t en \
    --batch-size 16 \
    --savedir ${SAVEDIR}
done


# Calculate Accuracy, replace the path to your location before run
python score.py ${SAVEDIR}/align \
    --langs ar,cs,de,en,es,fr,it,ja,ko,nl,ro,ru,tr,vi,zh

python score.py ${SAVEDIR}/noalign \
    --langs ar,cs,de,en,es,fr,it,ja,ko,nl,ro,ru,tr,vi,zh

