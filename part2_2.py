import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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


    y_prob = tf.matmul(X,W) + b

    # Error definition
    error =  tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_prob, labels=tf.one_hot(y_target, 10))  ), 
                                  name='mean_squared_error') + tf.multiply(tf.reduce_sum(tf.multiply(W, W)), lr/2)


    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
    train = optimizer.minimize(loss=error)
    return W, b, X, y_target, y_prob, error, train



def getAccuracy():

    W = tf.placeholder(tf.float32, [784, 10], name='input_w')
    b = tf.placeholder(tf.float32, shape = [1,10], name='input_b')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')

    y_target = tf.placeholder(tf.int32, [None,1], name='target_y')

    y_predicted = tf.nn.softmax(tf.matmul(X,W) + b)

    compare = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(tf.argmax(tf.one_hot(y_target, 10),1),1))

    accuracy = tf.reduce_mean(tf.cast(compare, tf.float32))

    return W, b, X, y_target, accuracy



lr = 0.005


trainDataSize = 15000
batchSize = 500


W, b, X, y_target, y_prob, error, train = buildGraph(lr)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

initialW = sess.run(W)
initialb = sess.run(b)

errors = []
epochs = []
accuracys = []

v_errors = []


a_W, a_b, a_X, a_target, a_accuracy = getAccuracy()



validData = np.reshape(GvalidData,[1000, 784])
validTarget = np.reshape(GvalidTarget,[1000, 1])


for step in range(1, 20000):

    trainData, trainTarget = GtrainData, GtrainTarget

    index = (step * batchSize)%trainDataSize
    
    trainData = np.reshape(trainData[index:index + batchSize],[batchSize, 784])
    trainTarget = np.reshape(trainTarget[index:index + batchSize],[batchSize, 1])

    _, err, currentW, currentb, yhat = sess.run([train, error, W, b, y_prob], feed_dict={X: trainData , y_target: trainTarget})

    errors.append(err)
    
    validData = np.reshape(GvalidData,[1000, 784])
    validTarget = np.reshape(GvalidTarget,[1000, 1])

    _, err, currentW, currentb, yhat = sess.run([train, error, W, b, y_prob], feed_dict={X: validData , y_target: validTarget})

    v_errors.append(err)
    
    epochs.append(step)

    # accuracy = sess.run([a_accuracy], feed_dict = {a_X: trainData, a_W: currentW, a_b: currentb, a_target: trainTarget})

    # accuracys.append(accuracy)


    if not step % (batchSize * 5):
        print("step - %d"%(step))



plt.figure(1)
plt.plot(epochs, v_errors,'-', label = "validation_errors")
plt.plot(epochs, errors,'-', label = "errors")
plt.legend()

plt.title("Linear Regression on mini batch with different learning rate")
plt.show()
































