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
""" Define SubwordEncoder base class. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class SubwordEncoder(object):
    """ Base class for encoding subwords. """

    def __init__(self):
        super(SubwordEncoder, self).__init__()

    @abstractmethod
    def encode(self, words, return_str=False):
        """ Encodes a sentence into subwords.

        Args:
            words: A string separated by spaces or a list of word tokens.
            return_str: True if the returned value is a string, otherwise a list of tokens.

        Returns: A list of subword tokens or a string.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, words, return_str=True):
        """ Recovers the result of `encode(words)`.

        Args:
            words: A string separated by spaces or a list of subword tokens.
            return_str: True if the returned value is a string, otherwise a list of tokens.

        Returns: The recovered sentence string or a list of tokens.
        """
        raise NotImplementedError
