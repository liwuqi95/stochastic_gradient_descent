import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np



with np.load("notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    GtrainData, GtrainTarget = Data[:15000], Target[:15000]
    GvalidData, GvalidTarget = Data[15000:16000], Target[15000:16000]
    GtestData, GtestTarget = Data[16000:], Target[16000:]



def buildGraph(lr):

    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784,10], stddev=0.5), name='weights')
    b = tf.Variable(tf.ones(shape=[1,10]), name='biases')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.int32, [None,1], name='target_y')


    y_prob = tf.nn.softmax(tf.matmul(X,W) + b)

    # Error definition
    error = -tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_prob, labels=tf.one_hot(y_target, 10))), 
                                  name='mean_squared_error') + tf.multiply(tf.reduce_sum(tf.multiply(W, W)), 0.01/2)

    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
    train = optimizer.minimize(loss=error)
    return W, b, X, y_target, y_prob, error, train





lrs = [0.005, 0.001, 0.0001]


trainDataSize = 15000
batchSize = 500

errors_array = []
epochs_array = []

randIndx = np.arange(trainDataSize)

for lr in lrs:

	W, b, X, y_target, y_prob, error, train = buildGraph(lr)

	init = tf.global_variables_initializer()
	sess = tf.InteractiveSession()
	sess.run(init)

	initialW = sess.run(W)
	initialb = sess.run(b)

	errors = []
	epochs = []

	for step in range(1, 200):

		np.random.shuffle(randIndx)

		trainData, trainTarget = GtrainData[randIndx], GtrainTarget[randIndx]
        
		trainData = np.reshape(trainData[:batchSize],[batchSize, 784])
		trainTarget = np.reshape(trainTarget[:batchSize],[batchSize, 1])


		_, err, currentW, currentb, yhat = sess.run([train, error, W, b, y_prob], feed_dict={X: trainData , y_target: trainTarget})

		errors.append(err)
		epochs.append(step)
		if not step % (batchSize * 5):
			print("step - %d"%(step))
	epochs_array.append(epochs)
	errors_array.append(errors)
	print("Final error for learning rate " + str(lr) + " is " +str(err))


































