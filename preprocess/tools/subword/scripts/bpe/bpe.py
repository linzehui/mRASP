#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use operations learned with learn_bpe.py to encode a new text.
The text will not be smaller, but use only a fixed vocabulary, with rare words
encoded as variable-length sequences of subword units.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""
from __future__ import unicode_literals, division

from .subword_encoder import SubwordEncoder
from .utils import force_to_str_list
from .utils import to_unicode
from .utils import PseudoPool
from .utils import load_protected_patterns
import argparse
import codecs
import sys
import re


class BPE(SubwordEncoder):
    def __init__(self, codes=None, separator=u'@@',
                 vocabulary=None, vocabulary_threshold=None):
        super(BPE, self).__init__()

        assert codes, "codes should be provided."
        codes = codecs.open(codes, encoding='utf-8')
        # check version information
        firstline = codes.readline()
        if firstline.startswith('#version:'):
            self.version = tuple([int(x) for x in re.sub(r'(\.0+)*$', '', firstline.split()[-1]).split(".")])
        else:
            self.version = (0, 1)
            codes.seek(0)

        self.bpe_codes = [tuple(item.split()) for item in codes]

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code, i) for (i, code) in reversed(list(enumerate(self.bpe_codes)))])
        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair, i in self.bpe_codes.items()])
        self.separator = separator
        if vocabulary:
            self.vocab = read_vocabulary(codecs.open(vocabulary, encoding="utf-8"), vocabulary_threshold)
        else:
            self.vocab = None
        self.glossaries = load_protected_patterns()

    def encode(self, words, return_str=False):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        try:
            tokens = words
            if not isinstance(words, list):
                tokens = words.strip().split()
            output = []
            for word in tokens:
                new_word = [out for segment in self._isolate_glossaries(word)
                            for out in bpe_encode(segment,
                                                  self.bpe_codes,
                                                  self.bpe_codes_reverse,
                                                  self.vocab,
                                                  self.separator,
                                                  self.version,
                                                  self.glossaries)]

                for item in new_word[:-1]:
                    output.append(item + self.separator)
                output.append(new_word[-1])
        except:
            print(words)
            raise Exception("string index out of range, string", tokens)
        if return_str:
            return " ".join(output)
        return output

    def decode(self, words, return_str=True):
        """ concatenates bpe subwords

        Args:
            words: A string separated by spaces or a list of subword tokens.

        Returns: The recovered sentence string.
        """
        sentence = words
        if isinstance(words, list):
            sentence = " ".join(words)
        sentence = sentence.replace(self.separator + u" ", u"")
        if sentence.endswith(u"@@"):
            sentence = sentence[:-2]
        if return_str:
            return sentence
        return force_to_str_list(sentence)

    def _isolate_glossaries(self, word):
        word_segments = [word]
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                             for out_segments in isolate_glossary(segment, gloss)]
        return word_segments


def get_pairs(word):
    """Return set of symbol pairs in a word.

    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def bpe_encode(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, glossaries=None, cache={}):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if orig in cache:
        return cache[orig]

    if orig in glossaries:
        cache[orig] = (orig,)
        return (orig,)

    if version == (0, 1):
        word = tuple(orig) + ('</w>',)
    elif version == (0, 2):  # more consistent handling of word-final segments
        word = tuple(orig[:-1]) + (orig[-1] + '</w>',)
    else:
        raise NotImplementedError

    pairs = get_pairs(word)

    if not pairs:
        return orig

    while True:
        bigram = min(pairs, key=lambda pair: bpe_codes.get(pair, float('inf')))
        if bigram not in bpe_codes:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>', ''),)

    if vocab:
        word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

    cache[orig] = word
    return word


def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher."""

    try:
        if final:
            left, right = bpe_codes[segment + '</w>']
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except:
        # sys.stderr.write('cannot split {0} further.\n'.format(segment))
        yield segment
        return

    if left + separator in vocab:
        yield left
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

    if (final and right in vocab) or (not final and right + separator in vocab):
        yield right
    else:
        for item in recursive_split(right, bpe_codes, vocab, separator, final):
            yield item


def check_vocab_and_split(orig, bpe_codes, vocab, separator):
    """Check for each segment in word if it is in-vocabulary,
    and segment OOV segments into smaller units by reversing the BPE merge operations"""

    out = []

    for segment in orig[:-1]:
        if segment + separator in vocab:
            out.append(segment)
        else:
            # sys.stderr.write('OOV: {0}\n'.format(segment))
            for item in recursive_split(segment, bpe_codes, vocab, separator, False):
                out.append(item)

    segment = orig[-1]
    if segment in vocab:
        out.append(segment)
    else:
        # sys.stderr.write('OOV: {0}\n'.format(segment))
        for item in recursive_split(segment, bpe_codes, vocab, separator, True):
            out.append(item)

    return out


def read_vocabulary(vocab_file, threshold):
    """read vocabulary file produced by get_vocab.py, and filter according to frequency threshold.
    """

    vocabulary = set()

    for line in vocab_file:
        word, freq = line.split()
        freq = int(freq)
        if threshold == None or freq >= threshold:
            vocabulary.add(word)

    return vocabulary


def isolate_glossary(word, glossary):
    """
    Isolate a glossary present inside a word.

    Returns a list of subwords. In which all 'glossary' glossaries are isolated 

    For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
        ['1934', 'USA', 'B', 'USA']
    """
    if word == glossary or glossary not in word:
        return [word]
    else:
        splits = word.split(glossary)
        segments = [segment.strip() for split in splits[:-1] for segment in [split, glossary] if segment != '']
        return segments + [splits[-1].strip()] if splits[-1] != '' else segments


def apply_fn(args):
    texts, cls_ins, recover_subword = args
    ret = []
    for text in texts:
        if recover_subword:
            ret.append(cls_ins.decode(to_unicode(text.strip()), return_str=True))
        else:
            ret.append(cls_ins.encode(to_unicode(text.strip()), return_str=True))
    return ret


def main(input, output, codes, recover_subword, n_threads, n_samples_per_thread, separator=u'@@'):
    bpe = BPE(codes, separator=separator)

    with PseudoPool(n_threads) as process_pool:
        sentence_list = []
        for line in input:
            sentence_list.append(line.strip())
            if len(sentence_list) >= n_samples_per_thread * n_threads:
                arg_list_list = PseudoPool.parse_arg_list(n_threads, sentence_list, bpe, recover_subword)
                # sys.stderr.write("loading {}...\n".format(len(sentence_list)))
                processed_list_list = process_pool.map(
                    apply_fn, arg_list_list)
                # sys.stderr.write("{} lines processed...\n".format(len(sentence_list)))
                for processed_list in processed_list_list:
                    for l in processed_list:
                        output.write(l + u"\n")
                del sentence_list[:]
                sentence_list = []
        if len(sentence_list) > 0:
            arg_list_list = PseudoPool.parse_arg_list(n_threads, sentence_list, bpe, recover_subword)
            processed_list_list = process_pool.map(
                apply_fn, arg_list_list)
            for processed_list in processed_list_list:
                for l in processed_list:
                    output.write(l + u"\n")
            del sentence_list[:]


if __name__ == "__main__":

    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="apply BPE")
    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH', help="Input text")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH', help="Output file")
    parser.add_argument(
        '--codes', '-c', type=str, metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--recover', '-r', action="store_true", default=False,
        help="Indicating the type: apply subword if False else recover the subword")
    parser.add_argument(
        '--threads', '-t', type=int,
        default=5, help="Num of threads")
    parser.add_argument(
        '--sample_per_thread', '-n', type=int, dest="sample_per_thread",
        default=20000, help="Num of samples per threads")

    args = parser.parse_args()

    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, "r", encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

    main(args.input, args.output, args.codes, args.recover, args.threads, args.sample_per_thread)
