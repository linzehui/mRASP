#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2018 ByteDance AI Lab, zhaochengqi.d@bytedance.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import unicode_literals, division
from __future__ import absolute_import

# pylint: disable=invalid-name
"""
Generate vocabulary from a pre-generated full vocabulary.
"""

import sys
import argparse
import collections
import logging
import codecs

if sys.version_info < (3, 0):
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
else:
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

parser = argparse.ArgumentParser(
    description="Generate vocabulary from a pre-generated full vocabulary.")
parser.add_argument(
    "--min_frequency",
    dest="min_frequency",
    type=int,
    default=0,
    help="Minimum frequency of a word to be included in the vocabulary.")
parser.add_argument(
    "--max_vocab_size",
    dest="max_vocab_size",
    type=int,
    help="Maximum number of tokens in the vocabulary.")
parser.add_argument(
    '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
    metavar='PATH', help="Input full vocabulary file.")
parser.add_argument(
    '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
    metavar='PATH', help="Output final vocabulary file")
args = parser.parse_args()

if args.input.name != '<stdin>':
    args.input = codecs.open(args.input.name, "r", encoding='utf-8')
if args.output.name != '<stdout>':
    args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

# Counter for all tokens in the vocabulary
cnt = collections.Counter()

for line in args.input:
    tokens = line.strip().split()
    cnt[tokens[0]] = int(tokens[1])

logging.info("Found %d unique tokens in the vocabulary.", len(cnt))

# Filter tokens below the frequency threshold
if args.min_frequency > 0:
    filtered_tokens = [(w, c) for w, c in cnt.most_common()
                       if c >= args.min_frequency]
    cnt = collections.Counter(dict(filtered_tokens))

logging.info("Found %d unique tokens with frequency > %d.",
             len(cnt), args.min_frequency)

# Sort tokens by 1. frequency 2. lexically to break ties
word_with_counts = cnt.most_common()
word_with_counts = sorted(
    word_with_counts, key=lambda x: (x[1], x[0]), reverse=True)

# Take only max-vocab
if args.max_vocab_size is not None and args.max_vocab_size > 0:
    word_with_counts = word_with_counts[:args.max_vocab_size]

for word, count in word_with_counts:
    args.output.write("{}\t{}\n".format(word, count))
