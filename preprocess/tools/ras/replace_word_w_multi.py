import os
import random
from collections import OrderedDict
from collections import defaultdict
import argparse
from pprint import pprint
import time
from tqdm import tqdm

random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--langs', required=True, help="The iso 639-2 code of languages that we apply RAS.")
parser.add_argument('--target-langs', required=True, help="The RAS target languages")
parser.add_argument('--multi_dict_path', required=True)
parser.add_argument('--data_path', required=True)
parser.add_argument('--num_repeat', type=int, default=1)
parser.add_argument('--max_dep', type=int, default=1)
parser.add_argument('--replace_prob', type=float, default=0.15)
parser.add_argument('--vocab_size', type=int, default=100000000)
parser.add_argument('--delimiter', type=str, default="__")

args = parser.parse_args()
pprint(args)


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
def load_multi_dict(dictionary_path, languages):
    """

    :param dictionary_path: the path the multi-way dictionary
    :param languages: languages that will be loaded to the final dict
    :return: a dictionary of dictionaries that stores all synonyms in `languages`
    """
    word_dict = {}
    with open(os.path.join(dictionary_path)) as f:
        i = 0
        for _line in f:
            _line = _line.strip()
            source = _line.split("\t")[0]
            src_lang = source[:2].lower()
            src_word = source[4:]
            if src_lang not in languages:
                continue  # skip languages that are not in `languages`
            if src_lang not in word_dict:
                word_dict[src_lang] = dict()
            if src_word not in word_dict[src_lang]:
                word_dict[src_lang][src_word] = OrderedDict()
            for word_str in _line.split("\t")[1:]:
                lang = word_str[:2].lower()
                word = word_str[4:-3]
                depth = word_str[-1]
                depth = int(depth)
                if depth not in word_dict[src_lang][src_word]:
                    word_dict[src_lang][src_word][depth] = []
                word_dict[src_lang][src_word][depth].append((word, lang))
            if i >= args.vocab_size:
                # only keep first `vocab_size` word pairs
                break
            i += 1
    
    return word_dict


# replace tokens in one sentence
def replace_one_sent(tokens, dictionary, target_langs):
    """

    :param tokens:
    :param dict: is a default dict with list as key
    :return:
    """
    cnt = 0
    new_tokens = []
    for token in tokens:
        _rep_token = None
        if token in dictionary:
            _dict = dictionary[token]
            _max_depth = min(len(_dict), args.max_dep)
            _candidates = []
            for dep in range(1, _max_depth + 1):
                # words with smaller depth has larger proba to be selected
                _candidates.extend(_dict[dep] * pow(2, _max_depth - dep))
            if random.random() < args.replace_prob:
                _iter = 0
                while _rep_token is None and _iter < 999:
                    _choice = random.choice(_candidates)
                    if _choice[1] in args.target_langs:
                        _rep_token = _choice[0]
                        cnt += 1
                    _iter += 1
        if _rep_token:
            new_tokens.append(_rep_token)
        else:
            new_tokens.append(token)
    
    return new_tokens, cnt


# from one sentence we get several copies
def replace_sent(sentence, dictionary, target_langs):
    """

    :param sentence: list of token in the sentence
    :param dictionary: dictionary of a certain language
    :return:
    """
    replace_cnt = 0
    total_token = 0
    replaced_sents = []
    # randomly select `num_repeat` languages from the list, create `num_repeat` copies for each sentence
    for _ in range(args.num_repeat):
        new_sent, cnt = replace_one_sent(sentence, dictionary, target_langs)
        if cnt > 0:
            replaced_sents.append(new_sent)
        replace_cnt += cnt
        total_token += len(new_sent)
    
    return replaced_sents, replace_cnt, total_token


if __name__ == "__main__":
    
    langs = [l for l in args.langs.split(";")]
    target_langs = [l for l in args.target_langs.split(";")]
    multi_dict_path = args.multi_dict_path
    data_path = args.data_path
    
    # 1. remove-bpe
    with open(os.path.join(data_path, "train.src"), 'r') as f, \
            open(os.path.join(data_path, "removed_bpe_file.src"), 'w+') as fw:
        for line in tqdm(f):
            new_line = remove_bpe(line, "@@ ")
            fw.write(new_line)
    
    print("======[Remove bpe Finished]======")
    
    # 2. load dictionaries
    dicts = load_multi_dict(multi_dict_path, langs)
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
            src_lang = src_sent[0][9:].lower()
            if src_lang in langs and src_lang in dicts:
                replaced_sents, replace_cnt, sent_token = replace_sent(sent, dicts[src_lang], target_langs)
                total_replace += replace_cnt
                total_token += sent_token
                replaced_sents = [sent for sent in replaced_sents]
                replaced_sents = [" ".join(sent) + "\n" for sent in replaced_sents]
                
                # write every sent so to save memory
                for r_sent in replaced_sents:
                    src_file_write.write(r_sent)
                
                for _ in range(len(replaced_sents)):
                    trg_file_write.write(trg_sent)
                    lang_indic_write.write(src_sent[0] + "\n")
            
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
