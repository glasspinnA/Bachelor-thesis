from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys

import tensorflow as tf
import numpy as np

_CSV_COLUMNS = ['PersonCount', 'Temperature', 'Humidity', 'Light', 'Pressure']

_CSV_COLUMN_DEFAULTS = [[0], [0.0], [0.0], [0.0], [0.0]]

_train_epochs = 200
_epochs_per_eval = 2
_batch_size = 40
_train_data = 'merged-data-randomised-train2.csv'
_test_data = 'merged-data-randomised-test2.csv'

_NUM_EXAMPLES = {
    'train': 21906,
    'validation': 9381, }


def build_model_columns():
  person_count = tf.feature_column.numeric_column('PersonCount')
  temperature = tf.feature_column.numeric_column('Temperature')
  humidity = tf.feature_column.numeric_column('Humidity')
  light = tf.feature_column.numeric_column('Light')
  pressure = tf.feature_column.numeric_column('Pressure')

  base_columns = [ temperature, humidity, light, pressure ]

  return base_columns


def build_estimator(model_dir, model_type):
    wide_columns = build_model_columns()

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
    return tf.estimator.LinearClassifier(model_dir=model_dir,feature_columns=wide_columns,config=run_config)

def input_fn(data_file, num_epochs, shuffle, batch_size):
  assert tf.gfile.Exists(data_file), (
      '%s not found.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    #features.pop('timestamp')
    label = features.pop('PersonCount')
    return features, tf.equal(label, 2)

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        shutil.rmtree('', ignore_errors=True)
        model = build_estimator('', '')

        for n in range(_train_epochs // _epochs_per_eval):
            model.train(input_fn=lambda: input_fn(
                _train_data, _epochs_per_eval, True, _batch_size))

            results = model.evaluate(input_fn=lambda: input_fn(
                _test_data, 1, False, _batch_size))

        # Display evaluation metrics
            print('Results at epoch', (n + 1) * _epochs_per_eval)
            print('-' * 60)

            for key in sorted(results):
                print('%s: %s' % (key, results[key]))
