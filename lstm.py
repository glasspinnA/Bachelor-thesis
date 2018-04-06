"""Human activity recognition using smartphones dataset and an LSTM RNN."""

# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

# The MIT License (MIT)
#
# Copyright (c) 2016 Guillaume Chevalier
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Also thanks to Zhao Yu for converting the ".ipynb" notebook to this ".py"
# file which I continued to maintain.

# Note that the dataset must be already downloaded for this script to work.
# To download the dataset, do:
#     $ cd data/
#     $ python download_dataset.py


import tensorflow as tf
import numpy as np



# Load "X" (the neural network's training and testing inputs)

def load_X(x_path):
    X_signals = []

    file = open(x_path, 'r')
    # Read dataset from disk, dealing with text files' syntax
    X_signals.append(
        [np.array(serie) for serie in [
            row.strip().split(',') for row in file
        ]]
    )
    #print(X_signals)
    file.close()

    return np.transpose(np.array(X_signals).astype(np.float), (1, 2, 0))


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.strip().split(' ') for row in file
        ]]
    ).astype(np.int32)
    print(y_)
    file.close()
    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing

    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # 33015 training series
        self.test_data_count = len(X_test)  # 14278 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series
        #print("n_steps:", len(X_train[0]))

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 400
        self.batch_size = 1500

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: 3 * 3D sensors features over time
        print("n_inputs: ", len(X_train[0][0]))
        self.n_hidden = 64  # nb of neurons inside the neural network
        self.n_classes = 18  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def LSTM_Network(_X, config):
    """Function returns a TensorFlow RNN with two stacked LSTM cells

    Two LSTM cells are stacked which adds deepness to the neural network.
    Note, some code of this notebook is inspired from an slightly different
    RNN architecture used on another dataset, some of the credits goes to
    "aymericdamien".

    Args:
        _X:     ndarray feature matrix, shape: [batch_size, time_steps, n_inputs]
        config: Config for the neural network.

    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.

      Args:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, config.n_inputs])
    # new shape: (n_steps*batch_size, n_input)

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, config.n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    #print("_X", _X)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]
    #print("last output", lstm_last_output)

    # Linear activation
    var = tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']
    #print(var)
    return var


def one_hot(y_):
    """
    Function to encode output labels from number indexes.

    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


if __name__ == "__main__":

    # -----------------------------
    # Step 1: load and prepare data
    # -----------------------------

    # Output classes to learn how to classify
    LABELS = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18"
    ]

    TRAIN_X = "merged-data-randomised-train-sensor.csv"
    TEST_X = "merged-data-randomised-test-sensor.csv"
    TRAIN_Y = "merged-data-randomised-train-pc.csv"
    TEST_Y = "merged-data-randomised-test-pc.csv"

    x_train_path = TRAIN_X
    x_test_path = TEST_X
    #print(x_test_path)
    #print(x_train_path)

    X_train = load_X(x_train_path)
    X_test = load_X(x_test_path)

    #print(X_train)

    y_train_path = TRAIN_Y
    y_test_path = TEST_Y
    y_train = one_hot(load_y(y_train_path))
    y_test = one_hot(load_y(y_test_path))

    #print(y_train)

    # -----------------------------------
    # Step 2: define parameters for model
    # -----------------------------------

    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_train.shape, y_train.shape,
          np.mean(X_train), np.std(X_train))
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    # ------------------------------------------------------
    # Step 3: Let's get serious and build the neural network
    # ------------------------------------------------------

    X = tf.placeholder(tf.float32, [None, config.n_steps, 1])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])

    pred_Y = LSTM_Network(X, config)
    #print("Y", Y)
    #print("pred_y", pred_Y)

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=pred_Y)) + l2
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    # --------------------------------------------
    # Step 4: Hooray, now train the neural network
    # --------------------------------------------

    # Note that log_device_placement can be turned ON but will cause console spam with RNNs.
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            #print(start, end)
            sess.run(optimizer, feed_dict={X: X_train[start:end],
                                           Y: y_train[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run(
            [pred_Y, accuracy, cost],
            feed_dict={
                X: X_test,
                Y: y_test
            }
        )
        print("traing iter: {},".format(i) +
              " test accuracy : {},".format(accuracy_out) +
              " loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")

    # ------------------------------------------------------------------
    # Step 5: Training is good, but having visual insight is even better
    # ------------------------------------------------------------------

    # Note: the code is in the .ipynb and in the README file
    # Try running the "ipython notebook" command to open the .ipynb notebook

    # ------------------------------------------------------------------
    # Step 6: And finally, the multi-class confusion matrix and metrics!
    # ------------------------------------------------------------------

    # Note: the code is in the .ipynb and in the README file
    # Try running the "ipython notebook" command to open the .ipynb notebook
