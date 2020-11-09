from Mykytea import Mykytea
import sys
import six
import re
import argparse

model = sys.argv[1]


def to_unicode(unicode_or_str):
    if six.PY3:
        return unicode_or_str
    if isinstance(unicode_or_str, str):
        value = unicode_or_str.decode('utf-8')
    else:
        value = unicode_or_str
    return value  # Instance of unicode


def force_to_unicode(s):
    """ Returns the joined string if s is a list. """
    if isinstance(s, list):
        s = " ".join(s)
    assert isinstance(s, six.string_types)
    return to_unicode(s)


def chinese_deseg(words):
    """ Recovers the result of `tokenize(words)`.

    Args:
        words: A list of strings, i.e. tokenized text.

    Returns: The recovered sentence string.
    """
    words = force_to_unicode(words)
    re_space = re.compile(r"(?<![a-zA-Z])\s(?![a-zA-Z])", flags=re.UNICODE)
    re_final_comma = re.compile("\.$")
    
    words = re_space.sub("", words)
    # words = words.replace(",", u"\uFF0C")
    words = re_final_comma.sub(u"\u3002", words)
    return words


class Kytea:
    def __init__(self, model):
        _param = u"-model {}".format(model)
        self._kt = Mykytea(_param)
    
    def tokenize(self, text):
        w_list = []
        for w in self._kt.getWS(text):
            w_list.append(w)
        return u" ".join(w_list)
    
    def detokenize(self, text):
        res = chinese_deseg(text.split(" "))
        res = re.sub(r" +", u" ", res)
        return res


def apply_fn(args):
    texts, cls_ins, recover_tok = args
    ret = []
    for text in texts:
        if recover_tok:
            ret.append(cls_ins.detokenize(to_unicode(text.strip())))
        else:
            ret.append(cls_ins.tokenize(to_unicode(text.strip())))
    return ret


def main(model, recover_tok):
    kt = Kytea(model=model)
    for line in sys.stdin:
        l = apply_fn(([line.strip()], kt, recover_tok))
        sys.stdout.write(l[0] + u"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="apply Kytea")
    parser.add_argument(
        '--recover', '-r', action="store_true", default=False,
        help="Indicating the type: apply subword if False else recover the subword")
    parser.add_argument(
        '--model', '-m', type=str, default="/opt/tiger/mrasp/kytea-0.4.7/data/model.bin",
        help="model file")
    args = parser.parse_args()
    main(args.model, args.recover)
