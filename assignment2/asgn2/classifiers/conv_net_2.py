import numpy as np

from asgn2.layers import *
from asgn2.fast_layers import *
from asgn2.layer_utils import *


class ConvNetTwo(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    C,H,W = input_dim
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    std = weight_scale
    self.params['W1'] = std * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = std * np.random.randn(num_filters*(H)*(W)/4, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = std * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    grads = {}
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    loss = 0
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    first_layer_output, first_layer_cache = conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
    conv_bnorm_firstlayer_op, conv_bnorm_cache = 


    sh = np.copy(first_layer_output)
    reshaped_first_x = first_layer_output.reshape((first_layer_output.shape[0], np.prod(first_layer_output.shape[1:])))
    second_layer_output, second_layer_cache = affine_relu_forward(reshaped_first_x,W2,b2)
    third_layer_output, third_layer_cache = affine_forward(second_layer_output,W3,b3)
    scores = np.copy(third_layer_output)
    if y is None:
      return scores
    
    loss, dout = softmax_loss(scores,y)
    reg = self.reg
    loss += (0.5 * reg * np.sum(W1*W1)) + (0.5 * reg * np.sum(W2*W2)) + (0.5 * reg * np.sum(W3*W3))

    reg = self.reg
    dthird_layer, grads['W3'], grads['b3']  = affine_backward(dout, third_layer_cache)
    dsecond_layer, grads['W2'], grads['b2']  = affine_relu_backward(dthird_layer, second_layer_cache)
    dfirst_layer, grads['W1'], grads['b1'] = conv_relu_pool_backward(dsecond_layer.reshape(sh.shape), first_layer_cache)
    grads['W3'] += reg * W3;
    grads['W2'] += reg * W2;
    grads['W1'] += reg * W1
    return loss,grads

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


    # loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################



    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
