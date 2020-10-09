import sys
from datetime import datetime


def get_dict_from_summary(logfile, skip_avg=True):
    """
    :param logfile: the summary.log file
    :param reversed: if True, rank by score from low to high
    :return: a dict of checkpoint_name: {testname: (score, time)}
    """
    log_dict = {}
    for line in logfile:
        time_str, ckpt_name, testname, score_str = line.split("\t")
        if skip_avg and "_avg_" in ckpt_name:
            continue
        time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        try:
            score = float(score_str)
        except Exception:
            continue
        if ckpt_name in log_dict:
            ckpt_dict = log_dict[ckpt_name]
        else:
            ckpt_dict = {}
        if testname in ckpt_dict:
            prev_time = ckpt_dict[testname][1]
            if time > prev_time:
                # update if have a newer score
                ckpt_dict[testname] = (score, time)
        else:
            ckpt_dict[testname] = (score, time)
        log_dict[ckpt_name] = ckpt_dict
    return log_dict


def get_ckpt_order(log_dict, testset, k=-1, reversed=False, print_score=False, no_print=False, print_with_pt=False):
    log_list = filter(lambda kv: testset in kv[1], log_dict.items())
    log_list = filter(lambda kv: kv[0] != 'checkpoint_best' and kv[0] != 'checkpoint_last', log_list)
    log_list = filter(lambda kv: kv[1][testset][0] is not None, log_list)
    def _get_num_from_ckptname(ckptname):
        _key = ckptname.split("_")[-1]
        assert _key.isdigit(), ("ckptname.split('_')[-1] is not digit!")
        return int(_key)
    log_list = sorted(log_list, key=lambda kv: (kv[1][testset][0], _get_num_from_ckptname(kv[0])), reverse=(not reversed))
    if k > 0 and k < len(log_list):
        log_list = log_list[:k]
    if not no_print:
        for ckpt, value in log_list:
            if print_score:
                print("{}\t{}".format(ckpt, value[testset][0]))
            else:
                if not print_with_pt:
                    print(ckpt)
                else:
                    print(f"{ckpt}.pt")
    return log_list


if __name__ == "__main__":
    _log_file = sys.stdin
    _testname = sys.argv[1]
    _topk = int(sys.argv[2]) if int(sys.argv[2]) > 0 else -1
    _print_score = eval(sys.argv[3])
    assert isinstance(_print_score, bool), TypeError('_print_score should be a bool')
    _skip_avg = eval(sys.argv[4])
    assert isinstance(_skip_avg, bool), TypeError('_skip_avg should be a bool')
    ret = get_dict_from_summary(_log_file, skip_avg=_skip_avg)
    get_ckpt_order(ret, _testname, k=_topk, print_score=_print_score)
