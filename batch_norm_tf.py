from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from util import get_normalized_data


def init_weight(M1, M2):
  return np.random.randn(M1, M2) * np.sqrt(2.0 / M1)


class HiddenLayerBatchNorm(object):
  def __init__(self, M1, M2, f):
    self.M1 = M1 # input size
    self.M2 = M2 # output size
    self.f = f # activation function 
    
    # parameter initailization 
    W = init_weight(M1, M2).astype(np.float32)
    gamma = np.ones(M2).astype(np.float32) # 1
    beta = np.zeros(M2).astype(np.float32) # 0
    
    # create tf variables 
    self.W = tf.Variable(W)
    self.gamma = tf.Variable(gamma)
    self.beta = tf.Variable(beta)

    # initailize the variables for test time 
    # trainable=False so the varaible won't be updated by gradient descent automatically  
    self.running_mean = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False) # 0
    self.running_var = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False) # 0

  def forward(self, X, is_training, decay=0.9): # decay varaible for updating the running_mean and running_variance 
    activation = tf.matmul(X, self.W) # linear transformation 
    if is_training:
      # tf.nn.moments calculates the mean and variance of x
      batch_mean, batch_var = tf.nn.moments(activation, [0])
      # tf.assign updates ref by assigning value to it.
      # create the update operations for the running_mean and running_variance 
      update_running_mean = tf.assign(
        self.running_mean, # The variable / ref you want to update
        self.running_mean * decay + batch_mean * (1 - decay) # The update expression 
      )
      update_running_var = tf.assign(
        self.running_var,
        self.running_var * decay + batch_var * (1 - decay)
      )
      
      # In order to make sure these updates get run every time we call train, we need to use tf.control_dependencies. It will have a list of Operation or Tensor objects which must be executed or computed before running the operations defined in the context. 
      # use tf.control_dependencies to make sure these updates are run whenever we do the tf.nn.batch_normalization operation. Then we ge the output of the BN. 
      with tf.control_dependencies([update_running_mean, update_running_var]):
        out = tf.nn.batch_normalization(
          activation, # X for BN
          batch_mean,
          batch_var,
          self.beta,
          self.gamma,
          1e-4
        )
    else: # for test
      out = tf.nn.batch_normalization(
        activation,
        self.running_mean,
        self.running_var,
        self.beta,
        self.gamma,
        1e-4
      )
    return self.f(out) # pass the BN out through the activation function 


# The regular hidden layer class 
class HiddenLayer(object):
  def __init__(self, M1, M2, f):
    self.M1 = M1
    self.M2 = M2
    self.f = f
    W = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)
    b = np.zeros(M2)
    self.W = tf.Variable(W.astype(np.float32))
    self.b = tf.Variable(b.astype(np.float32))

  def forward(self, X):
    return self.f(tf.matmul(X, self.W) + self.b)


class ANN(object):
  def __init__(self, hidden_layer_sizes):
    self.hidden_layer_sizes = hidden_layer_sizes

  def set_session(self, session):
    self.session = session

  def fit(self, X, Y, Xtest, Ytest, activation=tf.nn.relu, learning_rate=1e-2, epochs=15, batch_sz=100, print_period=100, show_fig=True):
    X = X.astype(np.float32)
    Y = Y.astype(np.int32)

    # initialize hidden layers
    N, D = X.shape
    self.layers = []
    M1 = D
    for M2 in self.hidden_layer_sizes:
      h = HiddenLayerBatchNorm(M1, M2, activation) # all the hidden layers are BN kind
      self.layers.append(h)
      M1 = M2
      
    # final layer
    K = len(set(Y))
    h = HiddenLayer(M1, K, lambda x: x) # the final layer is a regular hidden layer - a fully connected layer or a dense layer that has no activation function since TF implements the softmax later and internally. 
    self.layers.append(h)

    if batch_sz is None:
      batch_sz = N


    # note! we will need to build the output differently
    # for train and test (prediction)

    # set up theano functions and variables
    tfX = tf.placeholder(tf.float32, shape=(None, D), name='X')
    tfY = tf.placeholder(tf.int32, shape=(None,), name='Y')

    # for later use
    self.tfX = tfX

    # for training
    logits = self.forward(tfX, is_training=True)
    cost = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=tfY
      )
    )
    # train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    # train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.9).minimize(cost)
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True).minimize(cost)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # for testing
    test_logits = self.forward(tfX, is_training=False)
    self.predict_op = tf.argmax(test_logits, 1)

    # accuracy = tf.reduce_mean(1.0*(tfY == tf.argmax(logits, 1)))

    # init the variables
    self.session.run(tf.global_variables_initializer())

    n_batches = N // batch_sz
    costs = []
    for i in range(epochs):
      if n_batches > 1:
        X, Y = shuffle(X, Y)
      for j in range(n_batches):
        Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
        Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

        c, _, lgts = self.session.run([cost, train_op, logits], feed_dict={tfX: Xbatch, tfY: Ybatch})
        costs.append(c)
        if (j+1) % print_period == 0:
          acc = np.mean(Ybatch == np.argmax(lgts, axis=1))
          print("epoch:", i, "batch:", j, "n_batches:", n_batches, "cost:", c, "acc: %.2f" % acc)
          # print('dbg:', self.session.run(self.layers[0].running_mean).sum())

      print("Train acc:", self.score(X, Y), "Test acc:", self.score(Xtest, Ytest))
    
    if show_fig:
      plt.plot(costs)
      plt.show()

  # The forward is different for hidden layers and for activation layer 
  def forward(self, X, is_training):
    out = X
    for h in self.layers[:-1]:
      out = h.forward(out, is_training) 
    out = self.layers[-1].forward(out)
    return out

  def score(self, X, Y):
    P = self.predict(X)
    return np.mean(Y == P)

  def predict(self, X):
    return self.session.run(self.predict_op, feed_dict={self.tfX: X})


# Feedforwad neural network with batch normalization 
def main():
  # step 1: get the data and define all the usual variables
  Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

  ann = ANN([500, 300])

  # The only difference between Session and an InteractiveSession is that InteractiveSession makes itself the default session so that you can call run() or eval() without explicitly calling the session.
  session = tf.InteractiveSession()
  ann.set_session(session)

  ann.fit(Xtrain, Ytrain, Xtest, Ytest, show_fig=True)

  print("Train accuracy:", ann.score(Xtrain, Ytrain))
  print("Test accuracy:", ann.score(Xtest, Ytest))


if __name__ == '__main__':
  main()
