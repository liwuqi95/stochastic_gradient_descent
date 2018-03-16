import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = np.load("notMNIST.npz")

def getData():
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

def buildGraph(lr, decay):

    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784,1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')

    # Graph definition
    y_predicted = tf.matmul(X,W) + b

    # Error definition
    meanSquaredError = tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target), 
                                                reduction_indices=1, 
                                                name='squared_error'), 
                                  name='mean_squared_error') + tf.multiply(tf.reduce_sum(tf.multiply(W, W)), decay)

    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
    train = optimizer.minimize(loss=meanSquaredError)
    return W, b, X, y_target, y_predicted, meanSquaredError, train


def buildAccuracy():

    W = tf.placeholder(tf.float32, [784, 1], name='input_w')
    b = tf.placeholder(tf.float32, shape = (), name='input_b')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')

    y_predicted = tf.matmul(X,W) + b

    return W, b, X, y_target, y_predicted


decay_list = [0, 0.001, 0.1, 1]

trainDataSize = 3500

lr = 0.005
batchSize = 500

# global constants
GtrainData, GtrainTarget, GvalidData, GvalidTarget, GtestData, GtestTarget = getData()

randIndx = np.arange(trainDataSize)

for decay in decay_list:
    # Build computation graph
    W, b, X, y_target, y_predicted, meanSquaredError, train = buildGraph(lr, decay/2)

    errors = []
    epochs = []
    # Initialize session
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    initialW = sess.run(W)
    initialb = sess.run(b)


    for step in range(1, 20000):

        trainData, trainTarget = GtrainData, GtrainTarget

        index = (step * batchSize)%trainDataSize
        
        trainData = np.reshape(trainData[index:index + batchSize],[batchSize, 784])
        trainTarget = np.reshape(trainTarget[index:index + batchSize],[batchSize, 1])

        _, err, currentW, currentb, yhat = sess.run([train, meanSquaredError, W, b, y_predicted], feed_dict={X: trainData , y_target: trainTarget})

        if not step % (batchSize * 5):
                print("step - %d"%(step))

    a_W, a_b, a_X, a_y_target, a_y_predicted = buildAccuracy()


    validData = np.reshape(GvalidData,[100, 784])
    validTarget = np.reshape(GvalidTarget,[100, 1])

    y_pred = sess.run([a_y_predicted], feed_dict = {a_X: validData, a_y_target: validTarget, a_W: currentW, a_b: currentb})

    y_pred=np.reshape(y_pred,[100,1])

    compare_list = []
    for i in range(100):
        compare_list.append(validTarget[i] == (y_pred[i] > 0.5))

    accuracy = compare_list.count(True) / 100

    print("Valid Accuracy test for lumda "+ str(decay) +" is "+str(accuracy))


    testData = np.reshape(GtestData,[145, 784])
    testTarget = np.reshape(GtestTarget,[145, 1])

    y_pred = sess.run([a_y_predicted], feed_dict = {a_X: testData, a_y_target: testTarget, a_W: currentW, a_b: currentb})

    y_pred=np.reshape(y_pred,[145,1])

    compare_list = []
    for i in range(145):
        compare_list.append(testTarget[i] == (y_pred[i] > 0.5))

    accuracy = compare_list.count(True) / 145

    print("Valid Accuracy test for lumda "+ str(decay) +" is "+str(accuracy))






