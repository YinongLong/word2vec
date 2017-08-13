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
import tensorflow as tf


# Step 1: Download the data
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """
    首先判断数据文件是否存在，如果不存在，则会进行下载，然后判断文件的大小
    是否正确
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
    在这里读取数据的时候，仅使用压缩文件集中的第一个文件作为学习的数据
    """
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


# 这里的vocabulary是一个列表表示的单词序列（整个文件中的所有单词），根据这里的处理方式，
# 可以推测数据文件中的所有词汇是以空格间隔存储的。
vocabulary = read_data(filename)
# print('Type: ', type(vocabulary))
# print('Samples: ', vocabulary[:10])
# 进过输出可以看到，整个数据文件中有接近1700w的词汇
print('Data size', len(vocabulary))


# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words, n_words):
    """
    统计整个数据文件中单词出现的次数，并且仅选取出现次数最多的n_words个单词。
    生成词汇索引字典以及索引词汇字典的形式，并且根据词汇索引字典将整个数据集表
    示为索引表示，未出现在字典中的词汇使用'UNK'的索引值表示。
    返回索引表示的数据集，字典中出现的词汇在数据集中出现的次数，词汇索引字典，以及索引词汇字典。
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


# 全局变量，记录用来生成batch样本集的数据集索引位置
data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    """
    根据上面的词汇索引列表数据生成batch样本，其中skip_window表示以目标词汇为中心的滑动窗口两边
    词汇的个数，num_skips用来指定使用目标词汇来“想象”滑动窗口中的词汇个数（不包含目标词汇），而
    训练的每个样本由（目标词汇，滑动窗口中的某一个词汇）这样的形式构成。
    """
    global data_index
    assert batch_size % num_skips == 0
    # 确保“想象”的次数不能超过滑动窗口的大小
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [skip_window target skip_window]
    # 这里使用双端队列deque来存储滑动窗口中的词汇
    cus_buffer = collections.deque(maxlen=span)
    for _ in range(span):
        cus_buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        # 首先在“想象”的开始将目标词汇排除在外
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
    # 经过上面的过程，data_index的范围应该在0~len(data)-1内（包含端点），但是在这里我不是很清楚这样的处理过程到底
    # 是怎样一回事
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
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
np.random.seed(1992)  # Set random seed to repeat experiment.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                    stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch
    # tf.nce_loss automatically draws a new sample of the negative labels each time
    # we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))
    # Construct the SGD optimizer using a learning rate of 1.0
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer
    init = tf.global_variables_initializer()

# Step 5: Begin training
num_steps = 100001

with tf.Session(graph=graph) as session:
    # we must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run())
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0
        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)
except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
