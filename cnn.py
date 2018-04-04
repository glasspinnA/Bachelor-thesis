from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
import iris_data


def cnn(features,labels,mode):
	#Input layer = batch size, width, height, channels
	input_layer = tf.reshape(tf.cast(features["x"], tf.float32), [-1, 5, 5])


	#Första conv layer
	#Filter siffan kan ändras för att optimera modelen
	#Kernel size är storleken på varje filter
	conv1 = tf.layers.conv1d(
		inputs = input_layer,
		filters = 32,
		kernel_size = [3],
		padding = "same",
		activation = tf.nn.relu)

	#Pooling layer
	pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=[2], strides=[2])

	#Andra conv layer
	conv2 = tf.layers.conv1d(
		inputs = pool1,
		filters = 64,
		kernel_size = [3],
		padding="same",
		activation = tf.nn.relu)

	#Andra pool layer
	pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=[2], strides=[2])

	#Dense layers
	pool2_flat = tf.reshape(pool2, [-1, 4,4,4,4])
	
	dense = tf.layers.dense(
		inputs=pool2_flat, 
		units=512, 
		activation=tf.nn.relu)

	#Layer to help improve the result of the model
	dropout = tf.layers.dropout(
    inputs=dense, 
    rate=0.4, 
    training=mode == tf.estimator.ModeKeys.TRAIN)

	#Layer will return the raw values for our predictions
	#
	#154 antal output nerouner
	#
	#
	logits = tf.layers.dense(inputs=dropout, units=154)


	predictions = {
    	"classes": tf.argmax(input=logits, axis=1),
    	"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
  		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)	


  	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)



  	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
  		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  		train_op = optimizer.minimize(
        	loss=loss,
        	global_step=tf.train.get_global_step())
  		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


	eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
	#Load traning and test data
	(train_x, train_y),(test_x, test_y) = iris_data.load_data()
	#print(train_x)
	#print(train_y)
	#print(test_x)
	#print(test_y)
	#Create esimator 
	cnnEstimator = tf.estimator.Estimator(
		model_fn = cnn, 
		model_dir = "/tmp/convnet_model")

	tensorsLog = {"probabilities": "softmax_tensor"}
	logginHook = tf.train.LoggingTensorHook(
		tensors=tensorsLog, every_n_iter=50)

	#Train the model
	train_input = tf.estimator.inputs.numpy_input_fn(
  		x={"x": train_y},
      	y=train_y,
      	batch_size=100,
      	num_epochs=None,
      	shuffle=True)

	cnnEstimator.train(
    	input_fn=train_input,
    	steps=100,
    	hooks=[logginHook])

  	#Evaluate the model and print result
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	    x={"x": test_x},
	    y=test_y,
	    num_epochs=1,
	    shuffle=True) #False innan

	eval_results = cnnEstimator.evaluate(input_fn=eval_input_fn)
	print(eval_results)

if __name__ == '__main__':
	tf.app.run(main)
