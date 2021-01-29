from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
      scores = X[i].dot(W) # (C,)

      # shift value for 'scores' for numerical reasons
      scores -= scores.max() 

      probs = np.exp(scores)/np.sum(np.exp(scores))
      for j in range(num_class):
        # correct class
        if j == y[i]:
          loss += -np.log(probs[j])
          dW[:, j] += X[i].T.dot(probs[j] - 1)
        # incorrect classes
        else:
          dW[:, j] += X[i].T.dot(probs[j])
      
    # take average
    loss /= num_train
    dW /= num_train

    # add regularization
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]

    scores = X.dot(W) # (N, C)
    
    # shift scores for numerical stability
    scores -= scores.max(axis = 1, keepdims = True) # (N ,C)

    probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) # (N, C)

    # compute overall loss
    loss = -np.sum(np.log(probs[np.arange(num_train), y]))

    # compute overall gradient
    probs_grad = probs
    probs_grad[np.arange(num_train), y] -= 1
    dW = X.T.dot(probs_grad)

    # take average
    loss /= num_train
    dW /= num_train

    # add regularization
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
