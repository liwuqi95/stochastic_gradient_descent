import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as rd

#parameters
learning_rates = [0.005, 0.001, 0.0001]
weight_decay_coefficient = 0.01
iterations = 5000
batch_size = 500

def get_accuracy(predicted, true):
	predicted[predicted >= 0.5] = 1
	predicted[predicted < 0.5] = 0
	correct = 0.000000;
	count = true.shape[0]
	for i in range(count):
		if(predicted[i,0] == true[i,0]):
			correct = correct + 1
	return correct/count

#dataset
def load_data():
	with np.load("notMNIST.npz") as data :
		Data, Target = data ["images"], data["labels"]
		posClass = 2
		negClass = 9
		dataIndx = (Target==posClass) + (Target==negClass)
		Data = Data[dataIndx]/255.
		Target = Target[dataIndx].reshape(-1, 1)
		Target[Target==posClass] = 1
		Target[Target==negClass] = 0
		np.random.seed(521)
		randIndx = np.arange(len(Data))
		np.random.shuffle(randIndx)
		Data, Target = Data[randIndx], Target[randIndx]
		trainData, trainTarget = Data[:3500], Target[:3500]
		validData, validTarget = Data[3500:3600], Target[3500:3600]
		testData, testTarget = Data[3600:], Target[3600:]
		return trainData, trainTarget, validData, validTarget, testData, testTarget

trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()
trainData = trainData.reshape(trainData.shape[0], 28*28)
validData = validData.reshape(validData.shape[0], 28*28)

X = tf.placeholder(tf.float32, shape=(None, 28*28), name = 'inputs')
Y = tf.placeholder(tf.float32, shape=(None, 1), name = 'labels')
W = tf.Variable(tf.random_uniform([28*28, 1]), name = "weights")
b = tf.Variable(tf.zeros(1), name = "bias")
    
z = tf.add(tf.matmul(X, W), b)
prediction = tf.sigmoid(z, name = "prediction")

L_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=z))
L_W = weight_decay_coefficient * tf.nn.l2_loss(W)
total_loss = L_D + L_W

def train_1():
	for learning_rate in learning_rates:
		loss_list = list()
		accuracy_list = list()
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss = total_loss)
		session = tf.Session()
		session.run(tf.global_variables_initializer())
		for i in range(iterations):
			print("Iteration " + str(i+1) + " for a learning rate of " + str(learning_rate) + " starting...")
			start = rd.choice(range(3000))
			X_batch = trainData[start : start + batch_size]
			Y_batch = trainTarget[start : start + batch_size]
			print("Batch created...")
			train = session.run(optimizer, feed_dict={X: X_batch, Y: Y_batch})
			calculated_loss = session.run(total_loss, feed_dict={X: trainData, Y: trainTarget})
			predicted_y = session.run(prediction, feed_dict={X: trainData})
			accuracy = get_accuracy(predicted_y, trainTarget)
			loss_list.append(calculated_loss)
			accuracy_list.append(accuracy)
			print("Train cost: " + str(calculated_loss) + " after " + str(i+1) + " iterations. " + "Accuracy: " + str(accuracy))
		plt.plot(range(1, len(loss_list)+1), loss_list, '.')
		plt.show()
		plt.plot(range(1, len(accuracy_list)+1), accuracy_list, '.')
		plt.show()

def train_2():
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss = total_loss)
	session = tf.Session()
	session.run(tf.global_variables_initializer())
	loss_list = list()
	accuracy_list = list()
	for i in range(iterations):
		print("Iteration " + str(i+1) + " starting...")
		start = i % 3000
		X_batch = trainData[start : start + batch_size]
		Y_batch = trainTarget[start : start + batch_size]
		print("Batch created...")
		train = session.run(optimizer, feed_dict={X: X_batch, Y: Y_batch})
		calculated_loss = session.run(total_loss, feed_dict={X: trainData, Y: trainTarget})
		predicted_y = session.run(prediction, feed_dict={X: trainData})
		accuracy = get_accuracy(predicted_y, trainTarget)
		print("Train cost: " + str(calculated_loss) + " after " + str(i+1) + " iterations. " + "Accuracy: " + str(accuracy))
		loss_list.append(calculated_loss)
		accuracy_list.append(accuracy)
	plt.plot(range(1, len(loss_list)+1), loss_list, '.')
	plt.show()
	plt.plot(range(1, len(accuracy_list)+1), accuracy_list, '.')
	plt.show()

def part_2_1_3(iteration):
	feed = np.zeros(1)
	y_true = tf.zeros([1], tf.float32, name = 'label')
	y_hat = tf.placeholder(tf.float32, name = 'prediction')
	squared_error = tf.nn.l2_loss(y_hat)
	cross_entropy_error = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_hat)
	step = 1.0/iteration
	session = tf.Session()
	x_axis = list()
	y_axis = list()
	for i in range(iteration):
		x_axis.append(session.run(squared_error, feed_dict={y_hat: feed}))
		y_axis.append(session.run(cross_entropy_error, feed_dict={y_hat: feed}))
		feed[0] = feed[0] + step
	plt.plot(x_axis, y_axis, '.')
	plt.show()
train_1()
#part_2_1_3(1000)

