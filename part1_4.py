import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

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

def buildGraph():

    # Variable creation

    X = tf.placeholder(tf.float32, [None, 785], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')

    W = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(X), X)), tf.transpose(X)),y_target)

    # Graph definition
    y_predicted = tf.matmul(X,W)

    # Error definition
    meanSquaredError = tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target), 
                                                reduction_indices=1, 
                                                name='squared_error'), 
                                  name='mean_squared_error') 


    return W, X, y_target, y_predicted, meanSquaredError


trainDataSize = 3500

# Initialize session
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)



# global constants
GtrainData, GtrainTarget, GvalidData, GvalidTarget, GtestData, GtestTarget = getData()

start = time.time()

W, X, y_target, y_predicted, meanSquaredError = buildGraph()

trainData   = np.reshape(GtrainData,[trainDataSize, 784])
trainTarget = np.reshape(GtrainTarget, [trainDataSize, 1])


trainData = np.append(np.ones((trainDataSize, 1)), trainData, axis = 1)


err, currentW = sess.run([meanSquaredError, W], feed_dict={X: trainData , y_target: trainTarget})




print("Error is " + str(err))


validData = np.reshape(GvalidData,[100, 784])
validTarget = np.reshape(GvalidTarget,[100, 1])

validData = np.append(np.ones((100, 1)), validData, axis = 1)

yPredicted = tf.sigmoid(tf.matmul(tf.cast(validData, tf.float32),currentW))

error = tf.reduce_mean(tf.reduce_mean(tf.square(yPredicted - tf.cast(validTarget, tf.float32)), reduction_indices=1, name='squared_error'), name='mean_squared_error') 

print("Final Accuray is " + str(sess.run(1-error)))


print("Total time is " + str(time.time() - start))














