import numpy as np
from random import shuffle
import math


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_classes = W.shape[1]
  num_train = X.shape[0] 

  loss = 0.0

  for i in xrange(num_train):
    scores = X[i].dot(W)
    Vect = scores
    scores = scores - np.max(scores)
    total_sum = np.sum(np.exp(scores))
    correct_exp_score = np.exp(scores)[y[i]]
    Vect = np.exp(Vect) / total_sum # safe to do, gives the correct answer

    for j in range(num_classes):
      if(j==y[i]):
        loss+=(-1*np.log(Vect[j]))
        dW[:,y[i]] += (-1*(total_sum - correct_exp_score)/total_sum) * X[i]
      else:
        dW[:,j] += (1 * Vect[j]) * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  loss += 0.5*reg* np.sum(W*W)

  dW /= num_train
  dW += 0.5*reg* 2 * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  # NORMALIZER
  MaxArray_RowWise = np.max(scores,axis=1)
  MaxArray_RowWise = np.reshape(MaxArray_RowWise,(len(MaxArray_RowWise),1))
  scores = scores - MaxArray_RowWise
  exp_scores = np.exp(scores)
  total_exp_scores = np.sum(exp_scores, axis=1)
  normalizer = np.reshape(total_exp_scores,(len(total_exp_scores),1))
  M = exp_scores/normalizer
  dW += X.T.dot(M)

  loss = -1 * np.sum(np.log(M[range(len(M)),y]))
  loss/=num_train
  loss += 0.5*reg* np.sum(W*W)
  Mask = np.zeros_like(M)
  Mask[range(len(Mask)),y] = 1
  dW -= X.T.dot(Mask)
  dW/=num_train


  dW += 0.5 * reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

