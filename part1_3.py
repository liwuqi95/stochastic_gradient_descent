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

    # Initialize session
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    initialW = sess.run(W)
    initialb = sess.run(b)



    for step in range(1, 20000):   

        np.random.shuffle(randIndx)

        trainData, trainTarget = GtrainData[randIndx], GtrainTarget[randIndx]
        
        trainData = np.reshape(trainData[:batchSize],[batchSize, 784])
        trainTarget = np.reshape(trainTarget[:batchSize],[batchSize, 1])

        _, err, currentW, currentb, yhat = sess.run([train, meanSquaredError, W, b, y_predicted], feed_dict={X: trainData , y_target: trainTarget})


        if not step % (batchSize * 5):
                print("step - %d"%(step))


    validData = np.reshape(GvalidData,[100, 784])
    validTarget = np.reshape(GvalidTarget,[100, 1])

    yPredicted = tf.sigmoid(tf.matmul(tf.cast(validData, tf.float32),currentW)  + currentb)

    error = tf.reduce_mean(tf.reduce_mean(tf.square(yPredicted - tf.cast(validTarget, tf.float32)), reduction_indices=1, name='squared_error'), name='mean_squared_error') 

    print("Final Accuray for lumda is " + str(decay) + " is " + str(1-error))






















