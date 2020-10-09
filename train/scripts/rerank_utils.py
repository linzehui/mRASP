import re
import os
import sys
import argparse
from tqdm import tqdm


def _write_lines(itr, f=sys.stdout):
    def _flush_out(cached_list):
        f.writelines(cached_list)
        f.flush()
        return []
    cached_list = []
    for idx, r in enumerate(itr, start=1):
        if idx % 100000 == 0:
            cached_list = _flush_out(cached_list)
        cached_list.append(r)
    _flush_out(cached_list)


def remove_bpe(line, bpe_symbol="@@ "):
    line = line.replace("\n", '')
    line = (line + ' ').replace(bpe_symbol, '').rstrip()
    return line+("\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe", type=str, default="@@ ")
    return parser.parse_args()


def remove_bpe_fn(i=sys.stdin, o=sys.stdout, bpe="@@ "):
    lines = tqdm(i)
    lines = map(lambda x: remove_bpe(x, bpe), lines)
    _write_lines(lines, f=o)


def reprocess(fle, r2l=False):
    # takes in a file of generate.py translation generate_output
    # returns a source dict and hypothesis dict, where keys are the ID num (as a string)
    # and values and the corresponding source and translation. There may be several translations
    # per source, so the values for hypothesis_dict are lists.
    # parses output of generate.py

    r2l = False
    with open(fle, 'r') as f:
        txt = f.read()
    
    r2l = False
    """reprocess generate.py output"""
    p = re.compile(r"[STHP][-]\d+\s*")
    hp = re.compile(r"(\s*[-]?\d+[.]?\d+(e[+-]?\d+)?\s*)|(\s*(-inf)\s*)")
    source_dict = {}
    hypothesis_dict = {}
    score_dict = {}
    target_dict = {}
    pos_score_dict = {}
    lines = txt.split("\n")

    for line in lines:
        line += "\n"
        prefix = re.search(p, line)
        if prefix is not None:
            assert len(prefix.group()) > 2, "prefix id not found"
            _, j = prefix.span()
            id_num = prefix.group()[2:]
            id_num = int(id_num)
            line_type = prefix.group()[0]
            if line_type == "H":
                h_txt = line[j:]
                hypo = re.search(hp, h_txt)
                assert hypo is not None, ("regular expression failed to find the hypothesis scoring")
                _, i = hypo.span()
                score = hypo.group()
                hypo_str = h_txt[i:]
                if r2l:  # todo: reverse score as well
                    hypo_str = " ".join(reversed(hypo_str.strip().split(" ")))+"\n"
                if id_num in hypothesis_dict:
                    hypothesis_dict[id_num].append(hypo_str)
                    score_dict[id_num].append(float(score))
                else:
                    hypothesis_dict[id_num] = [hypo_str]
                    score_dict[id_num] = [float(score)]
            elif line_type == "S":
                source_dict[id_num] = (line[j:])
            elif line_type == "T":
                # target_dict[id_num] = (line[j:])
                continue
            elif line_type == "P":
                pos_scores = (line[j:]).split()
                pos_scores = [float(x) for x in pos_scores]
                if id_num in pos_score_dict:
                    pos_score_dict[id_num].append(pos_scores)
                else:
                    pos_score_dict[id_num] = [pos_scores]

    return source_dict, hypothesis_dict, score_dict, target_dict, pos_score_dict


def get_hypo_and_ref(fle, hyp_file, ref_input, ref_file, r2l=False, rank=0):
    with open(ref_input, 'r') as f:
        refs = f.readlines()
    _, hypo_dict, _, _, _ = reprocess(fle, r2l=r2l)
    assert rank < len(hypo_dict[0])
    maxkey = max(hypo_dict, key=int)
    f_hyp = open(hyp_file, "w")
    f_ref = open(ref_file, "w")
    for idx in range(maxkey+1):
        if idx not in hypo_dict:
            continue
        f_hyp.write(hypo_dict[idx][rank])
        f_ref.write(refs[idx])
    f_hyp.close()
    f_ref.close()


def recover_bpe(hyp_file):
    f_hyp = open(hyp_file, "r")
    f_hyp_out = open(hyp_file + ".nobpe", "w")
    for _s in ["hyp"]:
        f = eval("f_{}".format(_s))
        fout = eval("f_{}_out".format(_s))
        remove_bpe_fn(i=f, o=fout)
    f_hyp.close()
    f_hyp_out.close()


if __name__ == "__main__":
    filename = sys.argv[1]
    ref_in = sys.argv[2]
    model_type = sys.argv[3]
    if model_type == 'l2r':
        r2l = False
    elif model_type == "r2l":
        r2l = True
    else:
        raise Exception("`r2l_model` should be either 'l2r' or 'r2l'!")
    hypo_file = os.path.join(os.path.dirname(filename), "hypo.out")
    ref_out = os.path.join(os.path.dirname(filename), "ref.out")
    get_hypo_and_ref(filename, hypo_file, ref_in, ref_out, r2l=r2l)
    recover_bpe(hypo_file)
