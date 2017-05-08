# -*- coding: utf-8 -*-
"""
Created on 2017/5/8 下午9:14

@author: YinongLong

@file: word2vec_basic.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import math
import os
import random
import zipfile


import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf


# Step 1: Download the data
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """
    Download a file if not present, and make sure it's the right size.
    :param filename: 
    :param expected_bytes: 
    :return: 
    """
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify' + filename + '. Can you get to it with a browser?'
        )
    return filename


filename = maybe_download('./data/text8.zip', 31344016)


# Read the data into a list of strings.
def read_data(filename):
    """
    Extract the first file enclosed in a zip file as a list of words.
    :param filename: 
    :return: 
    """
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


vocabulary = read_data(filename)
# print('Type: ', type(vocabulary))
# print('Samples: ', vocabulary[:10])
print('Data size', len(vocabulary))


# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words, n_words):
    """
    Process raw inputs into a dataset.
    :param words: 
    :param n_words: 
    :return: 
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
del vocabulary
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    """
    :param batch_size: 
    :param num_skips: How many times to reuse an input to generate a label.
    :param skip_window: How many words to consider left and right.
    :return: 
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [skip_window target skip_window]
    cus_buffer = collections.deque(maxlen=span)
    for _ in range(span):
        cus_buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = cus_buffer[skip_window]
            labels[i * num_skips + j, 0] = cus_buffer[target]
        cus_buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


# Step 4: Build and train a skip-gram model.


batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.


# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by construction
# are also the most frequent.
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

def main():
    pass


if __name__ == '__main__':
    main()