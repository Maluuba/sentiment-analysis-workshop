#!/usr/bin/python
import argparse
import logging
import numpy as np
import os
import sys
import tensorflow as tf

from datetime import datetime

from embedding import glove, look_up_word, PAD_TOKEN

# Global constants
LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 16
EPOCHS = 300
MAX_LENGTH = 200
VALIDATION_SPLIT = 0.2
EMBEDDING_DIMS = glove.shape[1]
RNN_UNITS = 64
MIDDLE_DENSE_UNITS = RNN_UNITS


DATETIME_STRING = '{:%b-%d-%Y_%H%M%S}'.format(datetime.now())

# Sample usage:
# 1) train a model from scratch
#   python sentiment_rnn.py train --name experiment --base-dir /my/base/dir
#
# 2) continue training a model
#   python sentiment_rnn.py train --base-dir /my/base/dir/experiment --continue-epoch 2
#
# 3) runtimei from an epoch
#   python sentiment_rnn.py runtime --base-dir /my/base/dir/experiment --continue-epoch 2

# Parse cmd args
parser = argparse.ArgumentParser()
parser.add_argument('mode', help='`train` or `runtime`')
parser.add_argument('--name', default=None,
                    help='`Name of the job used as a prefix for identification')
parser.add_argument('--base-dir', default=os.path.dirname(__file__),
                    help='Where the model dir will be saved')
parser.add_argument('--continue-epoch', type=int, default=0,
                    help='If specified will start training model from this one')
parser.add_argument('--cache-dir', default=os.path.join(os.path.dirname(__file__), '.cache'),
                    help='Where the dataset and other intermediate files will be saved')
args = parser.parse_args()

start_epoch = args.continue_epoch
cache_dir = args.cache_dir
checkpoint_dir = args.base_dir

if args.mode == 'runtime':
    runtime = True
    logfile = 'runtime_%s.log' % DATETIME_STRING

elif args.mode == 'train':
    runtime = False
    logfile = 'train_%s.log' % DATETIME_STRING
    if args.name or start_epoch == 0:
        name = 'experiment_' + DATETIME_STRING if not args.name else args.name
        checkpoint_dir = os.path.join(checkpoint_dir, name)

else:
    raise RuntimeError('`%s` mode isn''t recognized!' % args.mode)


logfile = os.path.join(checkpoint_dir, logfile)

if not runtime and start_epoch == 0 and os.path.exists(checkpoint_dir):
    raise RuntimeError(
        'Model path exists (%s). Delete first or change the name to train from scratch.' % checkpoint_dir)

for d in [cache_dir, checkpoint_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(logfile),
        logging.StreamHandler()
    ]
)

logging.info('Downloading the polarity dataset...')
RELATIVE_POLARITY_DATASET_SUBDIR = os.path.join('datasets', 'review_polarity')
DATA_DIR = os.path.join(
    cache_dir, RELATIVE_POLARITY_DATASET_SUBDIR, 'txt_sentoken')
do_extract = not os.path.exists(DATA_DIR)

# Data retrieved from https://www.cs.cornell.edu/people/pabo/movie-review-data/
# Pang, B., & Lee, L. (2004, July). A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In Proceedings of the 42nd annual meeting on Association for Computational Linguistics (p. 271). Association for Computational Linguistics.

dataset = tf.keras.utils.get_file(
    fname='review_polarity.tar.gz',
    cache_dir=cache_dir,
    cache_subdir=RELATIVE_POLARITY_DATASET_SUBDIR,
    origin='https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz',
    extract=do_extract)

logging.info('Loading and preprocessing text data...')

train_file_sents, val_file_sents = [], []

for d, sent in [(os.path.join(DATA_DIR, sd), score) for sd, score in [('pos', 1.), ('neg', 0.)]]:
    files = os.listdir(d)
    split_index = int((1-VALIDATION_SPLIT)*len(files))
    train_file_sents += [(os.path.join(d, f), sent)
                         for f in files[:split_index]]
    val_file_sents += [(os.path.join(d, f), sent)
                       for f in files[split_index:]]


def make_token_generator_for_files(src_file_sents):
    def generator():
        for f, sent in src_file_sents:
            # Put your custom tokenizing code here
            # Use nltk.word_tokenize, but in this case the dataset is processed so we don't need to
            #   import nltk
            #   nltk.download('punkt')
            line_token_ids = [look_up_word(t.lower()) for ts in [line.split() for line in tf.gfile.GFile(
                f, 'r').readlines()] for t in ts][:MAX_LENGTH]
            token_ids_length = len(line_token_ids)
            # Could also do `padded_batch` here
            line_token_ids += [PAD_TOKEN] * (MAX_LENGTH - token_ids_length)
            yield (line_token_ids, token_ids_length, sent)
    return generator


logging.info('Making datasets...')

# We use a dynamic batch size so we can evaluate a whole dataset
batch_size = tf.placeholder(tf.int64)

train_set_size, val_set_size = len(train_file_sents), len(val_file_sents)

train_dataset = tf.data.Dataset.from_generator(
    make_token_generator_for_files(train_file_sents), (tf.int32, tf.int32, tf.float32), (tf.TensorShape([None]), tf.TensorShape(None), tf.TensorShape(None)))\
    .shuffle(train_set_size)\
    .batch(batch_size)

val_dataset = tf.data.Dataset.from_generator(
    make_token_generator_for_files(val_file_sents), (tf.int32, tf.int32, tf.float32), (tf.TensorShape([None]), tf.TensorShape(None), tf.TensorShape(None)))\
    .shuffle(val_set_size)\
    .batch(batch_size)

iterator = tf.data.Iterator.from_structure(
    train_dataset.output_types, train_dataset.output_shapes)

logging.info('Creating TensorFlow ops...')

# Create the dataset initialization operations
train_init_op = iterator.make_initializer(train_dataset)
val_init_op = iterator.make_initializer(val_dataset)

batch_token_ids, batch_seq_lens, batch_labels = iterator.get_next()

embedding_table = tf.get_variable("embedding_table", initializer=glove)

batch_embedding = tf.nn.embedding_lookup(embedding_table, batch_token_ids)

##### Actual model definition ###########

fwd = tf.contrib.rnn.GRUCell(num_units=RNN_UNITS)
bwd = tf.contrib.rnn.GRUCell(num_units=RNN_UNITS)

_, final_rnn_state = tf.nn.bidirectional_dynamic_rnn(
    fwd,
    bwd,
    batch_embedding,
    sequence_length=batch_seq_lens,
    dtype=tf.float32
)

# Final state of GRU is the same as final output. No need to slice outputs with tf.gather here.
fwd_state, bwd_state = final_rnn_state

last_rnn_state = tf.concat([fwd_state, bwd_state], axis=1)

sentiment_logits = tf.layers.dense(
    last_rnn_state,
    1,
    use_bias=True
)

# Training and Evaluation code follows

loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=sentiment_logits, labels=batch_labels))

accuracy = tf.metrics.accuracy(
    batch_labels,
    tf.greater(sentiment_logits, tf.zeros(tf.shape(sentiment_logits)))
)

optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
trainer = optimizer.minimize(loss_op)
saver = tf.train.Saver()

# A runtime method


def interact(sess):
    logging.info('='*40)
    logging.info('Interactive runtime')
    logging.info('='*40)
    inp = input('Enter a phrase or `q` to quit: ')
    while inp and inp != 'q':
        logging.info('Query: %s' % inp)
        line_token_ids = [look_up_word(t.lower())
                          for t in inp.split()][:MAX_LENGTH]
        token_ids_length = len(line_token_ids)
        line_token_ids += [PAD_TOKEN] * (MAX_LENGTH - token_ids_length)

        pred = sess.run([sentiment_logits], feed_dict={
            batch_token_ids: [line_token_ids], batch_seq_lens: [token_ids_length], batch_size: 1})

        if pred[0] >= 0:
            logging.info('Result: POSTIVE (+)')
        else:
            logging.info('Result: NEGATIVE (-)')

        inp = input('Enter a phrase or `q` to quit: ')


with tf.Session() as session:

    if runtime or start_epoch > 0:
        logging.info('Restoring model from %s' % checkpoint_dir)
        saver.restore(session, os.path.join(
            checkpoint_dir, 'model-{0}'.format(start_epoch)))
    else:
        logging.info('Training model from scratch in %s' % checkpoint_dir)
        session.run(tf.global_variables_initializer())

    session.run(tf.local_variables_initializer())
    session.run(tf.tables_initializer())

    if runtime:
        interact(session)
        sys.exit(0)

    for i in range(start_epoch + 1, start_epoch + EPOCHS + 1, 1):
        session.run(train_init_op, feed_dict={batch_size: TRAIN_BATCH_SIZE})
        logging.info('='*50)
        logging.info('EPOCH %d ' % i + '-'*40)
        # Iterate over batches
        batchn = 0
        while True:
            try:
                loss, acc, bs, _ = session.run(
                    [loss_op, accuracy, batch_size, trainer], feed_dict={batch_size: TRAIN_BATCH_SIZE})

                # Print stats at the start and end of the batch for debugging
                if batchn == 0 or batchn == ((train_set_size // bs) - 1):
                    logging.info('ep={}, batch={}, loss={:.5f}, acc={:.4f}'.format(
                        i, batchn, loss, acc[0]))

                batchn += 1

            except tf.errors.OutOfRangeError:
                break

        session.run(train_init_op, feed_dict={batch_size: train_set_size})
        loss, acc = session.run([loss_op, accuracy])
        logging.info('-'*50)
        logging.info(
            'TRAIN RESULTS: loss={:.5f}, acc={:.4f}'.format(loss, acc[0]))

        session.run(val_init_op, feed_dict={batch_size: val_set_size})
        loss, acc = session.run([loss_op, accuracy])
        logging.info(
            'VALIDATION RESULTS: loss={:.5f}, acc={:.4f}'.format(loss, acc[0]))

        saver.save(session, os.path.join(checkpoint_dir, 'model'), i)
