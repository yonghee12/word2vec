from typing import *

import numpy as np
import pandas as pd

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from functions import *

lemmatizer = WordNetLemmatizer()
lemm = lemmatizer.lemmatize

corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]


def get_tokenized_matrix_english(texts: List[str], lemmatize=True) -> List[List]:
    if lemmatize:
        return [[lemm(word) for word in word_tokenize(text)] for text in texts]
    else:
        return [word_tokenize(text) for text in texts]


def get_window_pairs(tokens: List[str], win_size=4) -> List[Tuple]:
    window_pairs = []
    for idx, token in enumerate(tokens):
        start = max(0, idx - win_size)
        end = min(len(tokens), idx + win_size + 1)
        for win_idx in range(start, end):
            if not idx == win_idx:
                window_pairs.append((token, tokens[win_idx]))
    return window_pairs


tokenized_matrix = get_tokenized_matrix_english(corpus, lemmatize=True)
unique_tokens = get_uniques_from_nested_lists(tokenized_matrix)
token2idx, idx2token = get_item2idx(unique_tokens)
vocab_size = len(token2idx)
print(idx2token.items())

pairs_matrix = [get_window_pairs(sent) for sent in tokenized_matrix]