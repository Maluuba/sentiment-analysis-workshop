#!/usr/bin/env python

# Mostly from https://github.com/Maluuba/qgen-workshop/blob/master/qgen/embedding.py
import os
import numpy as np
import tarfile

_word_to_idx = {}
_idx_to_word = []


def _add_word(word):
    idx = len(_idx_to_word)
    _word_to_idx[word] = idx
    _idx_to_word.append(word)
    return idx


UNKNOWN_WORD = "<UNK>"
PAD_WORD = ""

UNKNOWN_TOKEN = _add_word(UNKNOWN_WORD)
PAD_TOKEN = _add_word(PAD_WORD)

GLOVE_ZIPPED = os.path.join(os.path.dirname(__file__), 'glove.tar.gz')
GLOVE_UNZIPPED = os.path.join(os.path.dirname(__file__), 'glove.6B.50d.txt')


def look_up_word(word):
    return _word_to_idx.get(word, UNKNOWN_TOKEN)


def look_up_token(token):
    return _idx_to_word[token]


if not os.path.exists(GLOVE_UNZIPPED):
    tar = tarfile.open(GLOVE_ZIPPED, "r:gz")
    tar.extractall()
    tar.close()

embeddings_path = os.path.join(GLOVE_UNZIPPED)

with open(embeddings_path) as f:
    line = f.readline()
    chunks = line.split(" ")
    dimensions = len(chunks) - 1
    f.seek(0)

    vocab_size = sum(1 for line in f)
    vocab_size += 2
    f.seek(0)

    glove = np.ndarray((vocab_size, dimensions), dtype=np.float32)
    glove[UNKNOWN_TOKEN] = np.ones(dimensions)
    glove[PAD_TOKEN] = np.zeros(dimensions)

    for line in f:
        chunks = line.split(" ")
        idx = _add_word(chunks[0])
        glove[idx] = [float(chunk) for chunk in chunks[1:]]
        if len(_idx_to_word) >= vocab_size:
            break
