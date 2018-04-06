import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
#from sklearn.cross_validation import train_test_split

def read_data(file_path):
    column_names = ['timestamp', 'PersonCount', 'Temperature', 'Humidity', 'Light', 'Pressure']
    data = pd.read_csv(file_path,header = None, names = column_names)
    return data

def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)

def segment_signal(data, window_size = 150):
    segments = np.empty((0,window_size,4))
    labels = np.empty((0))
    #uniqueLabels = set()
    for (start, end) in windows(data['timestamp'], window_size):
        temperature = data["Temperature"][start:end]
        humidity = data["Humidity"][start:end]
        light = data["Light"][start:end]
        pressure = data["Pressure"][start:end]
        if(len(dataset['timestamp'][start:end]) == window_size):
            segments = np.vstack([segments, np.dstack([temperature, humidity, light, pressure])])
            #uniqueLabels.add(stats.mode(data["PersonCount"][start:end])[0][0])
            labels = np.append(labels, stats.mode(data["PersonCount"][start:end])[0][0])
    return segments, labels #np.array(list(uniqueLabels))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x,W, [1, 1, 1, 1], padding='VALID')

def apply_depthwise_conv(x,kernel_size,num_channels,depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights),biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')

dataset = read_data('merged-data-with-timestamp2.csv')

segments, labels = segment_signal(dataset)
print("Dummy:: ", pd.get_dummies(labels))
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)

reshaped_segments = segments.reshape(len(segments), 1, 150, 4)

print("ReshSegments::", reshaped_segments, len(reshaped_segments))
print("Segments::", segments)
print("Labels::", len(labels))

#train_x, test_x, train_y, test_y = train_test_split(reshaped_segments, labels, test_size=0.3, random_state=100)

train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
print("Split::", len(train_test_split))
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]

print("Train X::", len(train_x))
print("Test X::", len(test_x))
print("Train Y::", len(train_y))
print("Test Y::", len(test_y))

input_height = 1
input_width = 150
num_labels = 18
num_channels = 4

batch_size = 30
kernel_size = 5
depth = 10 #60
num_hidden = 64

learning_rate = 0.0001
training_epochs = 200

total_batches = train_x.shape[0]
print("Total batches::", total_batches)

X = tf.placeholder(tf.float32, shape=[None,input_height,input_width,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

c = apply_depthwise_conv(X,kernel_size,num_channels,depth)
p = apply_max_pool(c,20,2)
c = apply_depthwise_conv(p,6,depth*num_channels,depth//10)

shape = c.get_shape().as_list()
c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])

f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * num_channels * (depth//10), num_hidden])
f_biases_l1 = bias_variable([num_hidden])
f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1),f_biases_l1))

out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

loss = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)

with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(training_epochs):
        for b in range(total_batches):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            _, c = session.run([optimizer, loss],feed_dict={X: batch_x, Y : batch_y})
            cost_history = np.append(cost_history,c)
        print("Epoch: ",epoch," Training Loss: ",c," Training Accuracy: ", session.run(accuracy, feed_dict={X: train_x, Y: train_y}))

    print("Testing Accuracy:", session.run(accuracy, feed_dict={X: test_x, Y: test_y}))
