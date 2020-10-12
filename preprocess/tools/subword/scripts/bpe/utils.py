# -*- coding: utf-8 -*-
# from sacremoses

import io
import os
import re
import codecs
import json
import multiprocessing
import ahocorasick as ahc
from collections import Counter, defaultdict

import six

EMOJI_TRANS_PH = u'emoji_trans_ph{index}/>'
EMOJI_TRANS_PH_REGEX = u'emoji_trans_ph\d/>'
CONTPUNC_TRANS_PH = u'punc_trans_ph{index}/>'
CONTPUNC_TRANS_PH_REGEX = u'punc_trans_ph\d/>'
NO_TRANS_PH = u'no_trans_ph{index}/>'
NO_TRANS_PH_REGEX = u'no_trans_ph\d/>'
IGNORE_TRANS_PH = u'ignore_trans_ph{index}/>'
BREAKER_PH = u'no_trans_breaker/>'
MAX_RESERVED_SYMBOLS_PER_SENTENCE = 10
RESERVED_SYMBOLS = sum(
    [[ph.format(index=u"") if idx == MAX_RESERVED_SYMBOLS_PER_SENTENCE else ph.format(index=idx) \
      for idx in range(MAX_RESERVED_SYMBOLS_PER_SENTENCE + 1)]
     for ph in [EMOJI_TRANS_PH, IGNORE_TRANS_PH, NO_TRANS_PH, CONTPUNC_TRANS_PH]], [BREAKER_PH])

PROTECTED_PATTERNS_FILE = os.path.join(
    os.path.dirname(__file__),
    "resources/protected_patterns")
PROTECTED_PATTERNS = []
ALREADY_LOADED_PROTECTED_PATTERNS = False


def load_protected_patterns(
        protected_patterns_file=PROTECTED_PATTERNS_FILE):
    global ALREADY_LOADED_PROTECTED_PATTERNS
    global PROTECTED_PATTERNS
    if not ALREADY_LOADED_PROTECTED_PATTERNS:
        ALREADY_LOADED_PROTECTED_PATTERNS = True
        if os.path.exists(protected_patterns_file):
            with codecs.open(protected_patterns_file, "r", encoding="utf-8") as fp:
                PROTECTED_PATTERNS.extend([l.strip() for l in fp])
    return PROTECTED_PATTERNS


class EntityEncoder(object):
    __instances = dict()
    DICT_PATH = os.path.join(
        os.path.dirname(__file__), "resources/ne_dict")

    def __new__(cls, src_lang, trg_lang):
        lang_pair = "{}_{}".format(src_lang, trg_lang)
        if lang_pair not in EntityEncoder.__instances.keys():
            EntityEncoder.__instances[lang_pair] = super(
                EntityEncoder, cls).__new__(cls)
            EntityEncoder.__instances[lang_pair]._load_dict_file(src_lang, trg_lang)
            setattr(EntityEncoder.__instances[lang_pair], "_need_make_automaton", True)
        return EntityEncoder.__instances[lang_pair]

    def _load_dict_file(self, src_lang, trg_lang):
        lang_pair = "{}_{}".format(src_lang, trg_lang)
        lang1 = lang_pair.split("_")[0]
        file_path = os.path.join(EntityEncoder.DICT_PATH, lang_pair)
        self.dict_map = dict()
        self.automaton = None
        if os.path.exists(file_path):
            with codecs.open(file_path, "r", encoding="utf-8") as fp:
                for l in fp:
                    ent1, ent2, ent_type = l.strip("\n").split(";")
                    if lang1 == src_lang:
                        self.dict_map[ent1] = (ent2, ent_type)
                    else:
                        self.dict_map[ent2] = (ent1, ent_type)
            self.automaton = ahc.Automaton()
            if six.PY2:
                for k, v in self.dict_map.items():
                    self.automaton.add_word(k.lower().encode("utf-8"), (v, k.lower()))
            else:
                for k, v in self.dict_map.items():
                    self.automaton.add_word(k.lower(), (v, k.lower()))
            self.automaton.make_automaton()
            self._need_make_automaton = False

    def add_word(self, key, val, ent_type):
        if six.PY2:
            self.automaton.add_word(key.lower().encode("utf-8"), ((val, ent_type), key.lower()))
        else:
            self.automaton.add_word(key.lower(), ((val, ent_type), key.lower()))
        self._need_make_automaton = True

    def parse_entity(self, text, word_lc_list=None, percentage=0.9):
        if self._need_make_automaton:
            self.automaton.make_automaton()
            self._need_make_automaton = False
        if six.PY2:
            lower_text = text.lower().encode("utf-8")
            text = text.encode("utf-8")
        else:
            lower_text = text.lower()
        found_kw_lc = []
        for end_idx, (cat, keyword) in self.automaton.iter(lower_text):
            if six.PY2:
                begin_idx = end_idx - len(keyword.encode("utf-8")) + 1
                keyword = text[begin_idx:end_idx + 1].decode("utf-8")
            else:
                begin_idx = end_idx - len(keyword) + 1
                keyword = text[begin_idx:end_idx + 1]
            if len(found_kw_lc) == 0:
                found_kw_lc.append([keyword, cat, begin_idx, end_idx])
            else:
                if begin_idx > found_kw_lc[-1][-1]:
                    found_kw_lc.append([keyword, cat, begin_idx, end_idx])
                elif begin_idx > found_kw_lc[-1][-2]:  # overlap
                    pass
                else:  # replace
                    found_kw_lc[-1] = [keyword, cat, begin_idx, end_idx]
        tag_with_cnt = defaultdict(int)
        found_kws = dict()
        if six.PY2:
            text = text.decode("utf-8")
        for w in found_kw_lc:
            word_type = w[0]
            mapped_type = force_to_unicode(w[1][0])
            if w[1][1]:
                tag_type = w[1][1]
            else:
                tag_type = "PN"
            if word_lc_list is None or (
                    word_type in word_lc_list and
                    (word_lc_list.index(word_type)) / len(word_lc_list) > percentage):
                plhd = "<--plhd-{}-{}/>".format(tag_type, tag_with_cnt[tag_type])
                tag_with_cnt[tag_type] += 1
                found_kws[plhd] = mapped_type
                text = text.replace(word_type, plhd, 1)
        return text, found_kws


class Perluniprops:
    """
    This class is used to read lists of characters from the Perl Unicode
    Properties (see http://perldoc.perl.org/perluniprops.html).
    The files in the perluniprop.zip are extracted using the Unicode::Tussle
    module from http://search.cpan.org/~bdfoy/Unicode-Tussle-1.11/lib/Unicode/Tussle.pm
    """

    def __init__(self):
        self.datadir = os.path.dirname(os.path.abspath(__file__)) + '/resources/perluniprops/'
        # These are categories similar to the Perl Unicode Properties
        self.available_categories = ['Close_Punctuation', 'Currency_Symbol',
                                     'IsAlnum', 'IsAlpha', 'IsLower', 'IsN', 'IsSc',
                                     'IsSo', 'IsUpper', 'Line_Separator', 'Number',
                                     'Open_Punctuation', 'Punctuation', 'Separator',
                                     'Symbol']

    def chars(self, category=None, fileids=None):
        """
        This module returns a list of characters from  the Perl Unicode Properties.
        They are very useful when porting Perl tokenizers to Python.
            >>> from profanebleu.corpus import perluniprops as pup
            >>> pup.chars('Open_Punctuation')[:5] == [u'(', u'[', u'{', u'\u0f3a', u'\u0f3c']
            True
            >>> pup.chars('Currency_Symbol')[:5] == [u'$', u'\xa2', u'\xa3', u'\xa4', u'\xa5']
            True
            >>> pup.available_categories
            ['Close_Punctuation', 'Currency_Symbol', 'IsAlnum', 'IsAlpha', 'IsLower', 'IsN', 'IsSc', 'IsSo', 'IsUpper', 'Line_Separator', 'Number', 'Open_Punctuation', 'Punctuation', 'Separator', 'Symbol']
        :return: a generator of characters given the specific unicode character category
        """
        with io.open(self.datadir + category + '.txt', encoding='utf8') as fin:
            for ch in fin.read().strip():
                yield ch


class NonbreakingPrefixes:
    """
    This is a class to read the nonbreaking prefixes textfiles from the
    Moses Machine Translation toolkit. These lists are used in the Python port
    of the Moses' word tokenizer.
    """

    def __init__(self):
        self.datadir = os.path.dirname(os.path.abspath(__file__)) + '/resources/nonbreaking_prefixes/'
        self.available_langs = {'catalan': 'ca',
                                'czech': 'cs',
                                'german': 'de',
                                'greek': 'el',
                                'english': 'en',
                                'spanish': 'es',
                                'finnish': 'fi',
                                'french': 'fr',
                                'hungarian': 'hu',
                                'icelandic': 'is',
                                'italian': 'it',
                                'latvian': 'lv',
                                'dutch': 'nl',
                                'polish': 'pl',
                                'portuguese': 'pt',
                                'romanian': 'ro',
                                'russian': 'ru',
                                'slovak': 'sk',
                                'slovenian': 'sl',
                                'swedish': 'sv',
                                'tamil': 'ta'}
        # Also, add the lang IDs as the keys.
        self.available_langs.update({v: v for v in self.available_langs.values()})

    def words(self, lang=None, ignore_lines_startswith='#'):
        """
        This module returns a list of nonbreaking prefixes for the specified
        language(s).
            >>> from profanebleu.corpus import nonbreaking_prefixes as nbp
            >>> nbp.words('en')[:10] == [u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J']
            True
            >>> nbp.words('ta')[:5] == [u'\u0b85', u'\u0b86', u'\u0b87', u'\u0b88', u'\u0b89']
            True
        :return: a generator words for the specified language(s).
        """
        # If *lang* in list of languages available, allocate apt fileid.
        if lang in self.available_langs:
            filenames = ['nonbreaking_prefix.' + self.available_langs[lang]]
        # Use non-breaking praefixes for all languages when lang==None.
        elif lang == None:
            filenames = ['nonbreaking_prefix.' + v for v in
                         set(self.available_langs.values())]
        else:
            filenames = ['nonbreaking_prefix.en']

        for filename in filenames:
            with io.open(self.datadir + filename, encoding='utf8') as fin:
                for line in fin:
                    line = line.strip()
                    if line and not line.startswith(ignore_lines_startswith):
                        yield line


def is_cjk(character):
    """
    This checks for CJK character.
        >>> CJKChars().ranges
        [(4352, 4607), (11904, 42191), (43072, 43135), (44032, 55215), (63744, 64255), (65072, 65103), (65381, 65500), (131072, 196607)]
        >>> is_cjk(u'\u33fe')
        True
        >>> is_cjk(u'\uFE5F')
        False
    :param character: The character that needs to be checked.
    :type character: char
    :return: bool
    """
    return any([start <= ord(character) <= end for start, end in
                [(4352, 4607), (11904, 42191), (43072, 43135), (44032, 55215),
                 (63744, 64255), (65072, 65103), (65381, 65500),
                 (131072, 196607)]
                ])


control_chars_cf = r'[\xad\u0600\u0601\u0602\u0603\u0604\u0605\u061c\u06dd' \
                   r'\u070f\u08e2\u180e\u200b\u200c\u200d\u200e\u200f\u202a' \
                   r'\u202b\u202c\u202d\u202e\u2060\u2061\u2062\u2063\u2064' \
                   r'\u2066\u2067\u2068\u2069\u206a\u206b\u206c\u206d\u206e' \
                   r'\u206f\ufeff\ufff9\ufffa\ufffb\U000110bd\U0001bca0' \
                   r'\U0001bca1\U0001bca2\U0001bca3\U0001d173\U0001d174' \
                   r'\U0001d175\U0001d176\U0001d177\U0001d178\U0001d179' \
                   r'\U0001d17a\U000e0001\U000e0020\U000e0021\U000e0022' \
                   r'\U000e0023\U000e0024\U000e0025\U000e0026\U000e0027' \
                   r'\U000e0028\U000e0029\U000e002a\U000e002b\U000e002c' \
                   r'\U000e002d\U000e002e\U000e002f\U000e0030\U000e0031' \
                   r'\U000e0032\U000e0033\U000e0034\U000e0035\U000e0036' \
                   r'\U000e0037\U000e0038\U000e0039\U000e003a\U000e003b' \
                   r'\U000e003c\U000e003d\U000e003e\U000e003f\U000e0040' \
                   r'\U000e0041\U000e0042\U000e0043\U000e0044\U000e0045' \
                   r'\U000e0046\U000e0047\U000e0048\U000e0049\U000e004a' \
                   r'\U000e004b\U000e004c\U000e004d\U000e004e\U000e004f' \
                   r'\U000e0050\U000e0051\U000e0052\U000e0053\U000e0054' \
                   r'\U000e0055\U000e0056\U000e0057\U000e0058\U000e0059' \
                   r'\U000e005a\U000e005b\U000e005c\U000e005d\U000e005e' \
                   r'\U000e005f\U000e0060\U000e0061\U000e0062\U000e0063' \
                   r'\U000e0064\U000e0065\U000e0066\U000e0067\U000e0068' \
                   r'\U000e0069\U000e006a\U000e006b\U000e006c\U000e006d' \
                   r'\U000e006e\U000e006f\U000e0070\U000e0071\U000e0072' \
                   r'\U000e0073\U000e0074\U000e0075\U000e0076\U000e0077' \
                   r'\U000e0078\U000e0079\U000e007a\U000e007b\U000e007c' \
                   r'\U000e007d\U000e007e\U000e007f]'
control_chars_cc = r'[\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e' \
                   r'\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c' \
                   r'\x1d\x1e\x1f\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89' \
                   r'\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97' \
                   r'\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f]'


def normalize_punctuation(lang, text):
    text = re.sub(r'\r', r'', text)
    # text = re.sub(control_chars_cf, ' ', text)
    # text = re.sub(control_chars_cc, r' ', text)

    text = re.sub(r'\(', r' (', text)
    text = re.sub(r'\)', r') ', text)
    text = re.sub(r' +', r' ', text)
    text = re.sub(r'\) ([\.\!\:\?\;\,])', r')\1', text)
    text = re.sub(r'\( ', r'(', text)
    text = re.sub(r' \)', r')', text)
    text = re.sub(r'(\d) \%', r'\1%', text)
    text = re.sub(r' :', r':', text)
    text = re.sub(r' ;', r';', text)
    text = re.sub(r'\`', r"'", text)
    text = re.sub(r'\'\'', r' " ', text)
    text = re.sub(r'„', r'"', text)
    text = re.sub(r'“', r'"', text)
    text = re.sub(r'”', r'"', text)
    text = re.sub(r'–', r'-', text)
    text = re.sub(r'—', r' - ', text)
    text = re.sub(r' +', r' ', text)
    text = re.sub(r'´', r"'", text)
    text = re.sub(r'([a-zA-Z])‘([a-zA-Z])', r"\1'\2", text)
    text = re.sub(r'([a-zA-Z])’([a-zA-Z])', r"\1'\2", text)
    text = re.sub(r'‘', r'"', text)
    text = re.sub(r'‚', r'"', text)
    text = re.sub(r'’', r'"', text)
    text = re.sub(r"''", r'"', text)
    text = re.sub(r"´´", r'"', text)
    text = re.sub(r'…', r'...', text)
    # French quotes
    text = re.sub(r" « ", r' "', text)
    text = re.sub(r"« ", r'"', text)
    text = re.sub(r"«", r'"', text)
    text = re.sub(r" » ", r'" ', text)
    text = re.sub(r" »", r'"', text)
    text = re.sub(r"»", r'"', text)
    # handle pseudo-spaces
    text = re.sub(r' \%', r'%', text)
    text = re.sub(r'nº ', r'nº ', text)
    text = re.sub(r' :', r':', text)
    text = re.sub(r' ºC', r' ºC', text)
    text = re.sub(r' cm`', r' cm', text)
    text = re.sub(r' \?', r'?', text)
    text = re.sub(r' \!', r'!', text)
    text = re.sub(r' ;', r';', text)
    text = re.sub(r', ', r', ', text)
    text = re.sub(r' +', r' ', text)

    # English "quotation," followed by comma, style
    if lang == "en":
        text = re.sub(r'\"([,\.]+)', r'\1"', text)
    # Czech is confused
    elif lang == "cs" or lang == "cz":
        pass
    # German/Spanish/French "quotation", followed by comma, style
    else:
        text = re.sub(r',"', '",', text)
        text = re.sub(r'(\.+)"', r'"\1', text)
        # text = re.sub(r'(\.+)"(\s*[^<])', r'"\1\2', text)
    if lang == "de" or lang == "es" or lang == "cz" or lang == "cs" or lang == "fr":
        text = re.sub(r'(\d) (\d)', r'\1,\2', text)
    else:
        text = re.sub(r'(\d) (\d)', r'\1.\2', text)

    return text


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


def force_to_unicode(s):
    """ Returns the joined string if s is a list. """
    if isinstance(s, list):
        s = " ".join(s)
    assert isinstance(s, six.string_types)
    return to_unicode(s)


def force_to_str_list(s):
    """ Returns the token list of s. """
    if isinstance(s, six.string_types):
        s = to_unicode(s).strip().split()
    assert isinstance(s, list)
    return s


def to_unicode(unicode_or_str):
    if six.PY3:
        return unicode_or_str
    if isinstance(unicode_or_str, str):
        value = unicode_or_str.decode('utf-8')
    else:
        value = unicode_or_str
    return value  # Instance of unicode


def to_string_type(unicode_or_str):
    if six.PY3:
        return unicode_or_str
    if isinstance(unicode_or_str, str):
        value = unicode_or_str
    else:
        value = unicode_or_str.encode("utf-8")
    return value


def get_vocabulary(fobj, is_dict_style=False):
    """Read text and return dictionary that encodes vocabulary
    """
    vocab = Counter()
    for line in fobj:
        if is_dict_style:
            word, count = line.strip().split()
            vocab[word] = int(count)
        else:
            vocab.update([_ for _ in line.strip().split() if len(_) > 0])
    return vocab


class PseudoPool(object):
    def __init__(self, processes=1):
        """ If processes is 1, then don't create pool.

        Args:
            processes:
        """
        self.pool = None
        if processes > 1:
            self.pool = multiprocessing.Pool(processes=processes)
        self.processes = processes

    @staticmethod
    def parse_arg_list(n_threads, sample_list, *args):
        total_length = len(sample_list)
        samples_per_thread = total_length // n_threads + 1
        if samples_per_thread < 1000:
            samples_per_thread = total_length
        start_idx = 0
        ret = []
        while start_idx < total_length:
            arg_list = []
            arg_list.append(sample_list[start_idx:(start_idx + samples_per_thread)])
            for arg in args:
                if isinstance(arg, list) and len(arg) == len(sample_list):
                    arg_list.append(arg[start_idx:(start_idx + samples_per_thread)])
                elif not isinstance(arg, list):
                    arg_list.append(arg)
                else:
                    raise ValueError
            start_idx += samples_per_thread
            ret.append(arg_list)
        return ret

    def map(self, func, args_list):
        return [func(args) for args in args_list]

    def __enter__(self):
        if self.processes > 1:
            return self.pool
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.processes > 1:
            self.pool.terminate()


def text_encoders_yaml2json(loader_dict, output_folder, plain_text_extractor):
    with open(PROTECTED_PATTERNS_FILE, "r") as fp:
        protected = [l.strip() for l in fp]

    subword_encoder_mapping = {
        'TransformerSubword': 'TransformerSubwordEncoder',
        'BPE': 'BytePairEncoder'}

    src_dict = loader_dict['source_text_encoder']
    trg_dict = loader_dict['target_text_encoder']

    src_lang = src_dict['language']
    trg_lang = trg_dict['language']
    src_transformers = [('DataGate', {'lang': src_lang})]
    trg_transformers = [('DataGate', {'lang': trg_lang})]

    for xx_transformers, xx_dict in zip([src_transformers, trg_transformers],
                                        [src_dict, trg_dict]):
        if 'normalize_punctuation' in xx_dict:
            xx_transformers.append(
                ('PunctuationNormalizer', {'lang': xx_dict['language']}))
        xx_transformers.append(
            ('UnescapeSpecChars', {'lang': xx_dict['language']}))
        if 'tokenizer.class' in xx_dict:
            tkwargs = {}
            if 'tokenizer.params' in xx_dict:
                tkwargs = xx_dict['tokenizer.params']
            if xx_dict['tokenizer.class'] != "TransformerTokenizer":
                tkwargs['lang'] = xx_dict['language']
            if xx_dict['tokenizer.class'] in ["TransformerTokenizer", "MosesTokenizer"]:
                tkwargs["glossaries"] = protected
            if xx_dict['tokenizer.class'] == "ByteSegmentationNew":
                tkwargs["dict_path"] = "/mnt/labmtmodel/byteseg_v2/chengqi/dict"
            elif xx_dict["tokenizer.class"] == "PyKytea":
                tkwargs["model"] = "/opt/tiger/mt_serving/kytea_model.bin"
            xx_transformers.append((xx_dict['tokenizer.class'], tkwargs))

        if 'truecase_model' in xx_dict and xx_dict["truecase_model"]:
            if xx_dict is src_dict:
                with open(xx_dict['truecase_model']) as f:
                    xx_transformers.append(
                        ('MosesTruecaser',
                         {'model_str': f.read(), 'lang': xx_dict['language']}))
            else:
                xx_transformers.append(
                    ('MosesTruecaser',
                     {'model_str': '', 'lang': xx_dict['language']}))

        if 'subword_encoder.class' in xx_dict:
            subword_encoder = subword_encoder_mapping[
                xx_dict['subword_encoder.class']]
            if subword_encoder == 'BytePairEncoder':
                subword_encoder_codes = xx_dict['subword_encoder.params'][
                    'codes']
                with open(subword_encoder_codes) as f:
                    xx_transformers.append((
                        subword_encoder, {'codes_str': f.read(), 'glossaries': protected}))
            elif subword_encoder == 'TransformerSubwordEncoder':
                subword_encoder_vocab = xx_dict['subword_encoder.params'][
                    'subword_vocab']
                with open(subword_encoder_vocab) as f:
                    xx_transformers.append((
                        subword_encoder, {'file_str': f.read(), 'glossaries': protected}))

    with open(src_dict['vocabulary']) as f:
        src_vocab = f.readlines()
    with open(trg_dict['vocabulary']) as f:
        trg_vocab = f.readlines()

    src_vocab_words = " ".join([
        x.strip().split()[0] for x in src_vocab])
    empty_token = "EMPTY_TOKEN"
    while empty_token in src_vocab_words:
        empty_token += "0"

    mappers = {'src_ids': [('SymbolsMapper',
                            {'tokens': src_vocab,
                             'embedding_dim': -1,
                             'att_token': '',
                             'empty_token': empty_token,
                             'max_len': 255})],
               'trg_ids': [('SymbolsMapper',
                            {'tokens': trg_vocab,
                             'embedding_dim': -1,
                             'att_token': '',
                             'empty_token': '',
                             'max_len': 255})]}
    rectifiers = [
        ('RegexBreaker',
         {'regex_pattern_string': 'EMOJI_REGEXP_PATTERN_STRING', 'lang': src_lang, 'regex_from_etl': True})]
    if trg_lang in ["zh", "ja", "ko"]:
        rectifiers.append(
            ("ChineseSciformChecker", {}))

    if plain_text_extractor == "basic":
        rectifiers.extend([
            ('RegexBreaker',
             {'regex_pattern_string': 'JA_KAOMOJI_REGX', 'lang': src_lang, 'regex_from_etl': True}),
            # ('RegexBreaker',
            #  {'regex_pattern_string': 'SIMPLE_KAOMOJI_REGEX', 'lang': src_lang, 'regex_from_etl': True}),
            ('RegexBreaker',
             {'regex_pattern_string': 'LARK_EMOJI_REGEX', 'lang': src_lang, 'regex_from_etl': True}),
            ('UrlPuncBreaker',
             {'replace_url': True, 'translate_to': trg_lang})])

    elif plain_text_extractor == "ne_plhd":
        rectifiers.extend([
            # ('RegexBreaker',
            #  {'regex_pattern_string': 'JA_KAOMOJI_REGX', 'lang': src_lang, 'regex_from_etl': True}),
            # ('RegexBreaker',
            #  {'regex_pattern_string': 'SIMPLE_KAOMOJI_REGEX', 'lang': src_lang, 'regex_from_etl': True}),
            ('RegexBreaker',
             {'regex_pattern_string': 'LARK_EMOJI_REGEX', 'lang': src_lang, 'regex_from_etl': True}),
            ('UrlPuncBreaker',
             {'replace_url': False, 'translate_to': trg_lang})])
        word_mappings = []
        lang_pair = "{}_{}".format(src_lang, trg_lang)
        lang1 = lang_pair.split("_")[0]
        file_path = os.path.join(EntityEncoder.DICT_PATH, lang_pair)
        if os.path.exists(file_path):
            with codecs.open(file_path, "r", encoding="utf-8") as fp:
                for l in fp:
                    ent1, ent2, ent_type = l.strip("\n").split(";")
                    if lang1 == src_lang:
                        word_mappings.append((ent1, ent2, ent_type))
                    else:
                        word_mappings.append((ent2, ent1, ent_type))
        rectifiers.append(
            ('NameEntityBreaker',
             {'lang': src_lang, 'enable_lark': True, 'word_mappings': word_mappings}))

    data_loader_params = {
        'src_transformers': src_transformers,
        'trg_transformers': trg_transformers,
        'mappers': mappers,
    }
    if rectifiers and len(rectifiers) > 0:
        data_loader_params["rectifiers"] = rectifiers

    # os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'data_loader_params.json'),
              'w') as jp:
        json.dump(data_loader_params, jp, indent=4)
