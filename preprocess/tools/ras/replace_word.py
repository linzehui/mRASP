import os
import random
from collections import defaultdict
import argparse
from pprint import pprint
import time

random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--langs', required=True, help="The iso 639-2 code of languages that we apply RAS."
                                                   " We use MUSE dictionary for each en-x pair.")
parser.add_argument('--dict_path', required=True)
parser.add_argument('--data_path', required=True)
parser.add_argument('--num_repeat', type=int, default=1)
parser.add_argument('--replace_prob', type=float, default=0.15)
parser.add_argument('--vocab_size', type=int, default=2000)
args = parser.parse_args()
pprint(args)

langs = [l for l in args.langs.split(";") if l != "en"]
dict_path = args.dict_path
data_path = args.data_path


# remove bpe
def remove_bpe(input_sentence, bpe_symbol="@@ "):
    """
    
    :param input_sentence: the input sentence that is processed by bpe sub-word operation.
    :param bpe_symbol: the bpe indicator(default: '@@ '), which will be removed after the `remove_bpe` function
    :return: the output sentence that is recovered from  sub-word operation.
    """
    _sentence = input_sentence.replace("\n", '')
    _sentence = (_sentence + ' ').replace(bpe_symbol, '').rstrip()
    return _sentence + "\n"


# read dictionaries
def load_dicts(dictionary_path, languages):
    """
    
    :param dictionary_path: the path of the root of MUSE dictionaries
    :param languages: en-x dictionaries will be used for all x in `languages`
    :return: a dictionary of dictionaries that stores all word pairs for all en-x in `languages`
    """
    dict_of_dict = {}
    for lang in languages:
        _pair_name = ["en", "-", lang]
        pair_name = "".join(_pair_name)
        if not os.path.isfile(os.path.join(dictionary_path, pair_name+".txt")):
            raise FileNotFoundError("{}/{} not exists!".format(dictionary_path, pair_name+".txt"))
        x2y_dict = defaultdict(list)
        with open(os.path.join(dictionary_path, pair_name+".txt")) as f:
            i = 0
            for _line in f:
                bi_text = _line.strip().split()
                assert len(bi_text) == 2, ("in file {}/{}, line index {} has an invalid number of columns {}"
                                           .format(dictionary_path, pair_name, i, len(bi_text)))
                x2y_dict[bi_text[0]].append(bi_text[1])
                if i >= args.vocab_size:
                    # only keep first `vocab_size` word pairs
                    break
                i += 1
        dict_of_dict[pair_name] = x2y_dict

    return dict_of_dict


# replace tokens in one sentence
def replace_one_sent(tokens, dictionary):
    """

    :param tokens:
    :param dict: is a default dict with list as key
    :return:
    """
    cnt = 0
    new_tokens = []
    for token in tokens:
        if token in dictionary and random.random() < args.replace_prob:
            new_tokens.append(random.choice(dictionary[token]))
            cnt += 1
        else:
            new_tokens.append(token)

    return new_tokens, cnt


# from one sentence we get several copies
def replace_sent(sentence, dictionaries):
    """

    :param sentence: list of token in the sentence
    :param dictionaries: all dictionaries
    :param langs: all languages involved
    :return:
    """
    replace_cnt = 0
    total_token = 0
    replaced_sents = []
    selected_langs = random.sample(langs, args.num_repeat)
    # randomly select `num_repeat` languages from the list, create `num_repeat` copies for each sentence
    for _lang in selected_langs:
        dict_name = "en-" + _lang
        assert dict_name in dictionaries, ("{} not in dictionaries!".format(dict_name))
        selected_dict = dictionaries[dict_name]
        new_sent, cnt = replace_one_sent(sentence, selected_dict)
        if cnt > 0:
            replaced_sents.append(new_sent)
        replace_cnt += cnt
        total_token += len(new_sent)

    return replaced_sents, replace_cnt, total_token


if __name__ == "__main__":
    # 1. remove-bpe
    with open(os.path.join(data_path, "train.src"), 'r') as f, \
            open(os.path.join(data_path, "removed_bpe_file.src"), 'w+') as fw:
        for line in f:
            new_line = remove_bpe(line, "@@ ")
            fw.write(new_line)
    
    print("======[Remove bpe Finished]======")
    
    # 2. load dictionaries
    dicts = load_dicts(dict_path, langs)
    print(dicts.keys())
    
    for dict_name in dicts:
        print("The length of dict {dict_name} is {len}".format(dict_name=dict_name, len=len(dicts[dict_name])))
    print("======[Dicts Loaded]======")
    
    start_time = time.time()
    # 3. replace
    with open(os.path.join(data_path, 'removed_bpe_file.src'), 'r') as src_file_read, \
            open(os.path.join(data_path, 'expanded_train.src'), 'w+') as src_file_write, \
            open(os.path.join(data_path, "train.trg"), 'r') as trg_file_read, \
            open(os.path.join(data_path, "expanded_train.trg"), 'w+') as trg_file_write, \
            open(os.path.join(data_path, "lang_indicator.src"), 'w+') as lang_indic_write:
        total_replace, total_token = 0, 0
        src_sent = src_file_read.readline()
        trg_sent = trg_file_read.readline()
        while src_sent and trg_sent:
            src_sent = src_sent.strip().split()
            sent = src_sent[1:]  # remove language token
            
            replaced_sents, replace_cnt, sent_token = replace_sent(sent, dicts)
            total_replace += replace_cnt
            total_token += sent_token
            replaced_sents = [sent for sent in replaced_sents]
            replaced_sents = [" ".join(sent) + "\n" for sent in replaced_sents]
    
            # write every sent so to save memory
            for r_sent in replaced_sents:
                src_file_write.write(r_sent)
    
            for _ in range(len(replaced_sents)):
                trg_file_write.write(trg_sent)
                lang_indic_write.write(src_sent[0]+"\n")
    
            src_sent = src_file_read.readline()
            trg_sent = trg_file_read.readline()
    
    print("======[Replaced with dict Finished]======")
    print("Done in {} seconds".format(time.time() - start_time))
    
    print("Total Tokens(with repeated times) is {total_token}, with {replaced_token} replaced.\n"
          "With a proportion of {proportion}% \n"
          "The repeated times are set to {num_repeat}"
          .format(total_token=total_token, replaced_token=total_replace, num_repeat=args.num_repeat,
                  proportion=total_replace / total_token * 100))

    os.system("rm {}".format(os.path.join(data_path, 'removed_bpe_file.src')))
