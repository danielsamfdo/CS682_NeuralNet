import numpy as np
import copy as copy

"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  next_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + b)
  cache = (x,Wx,prev_h,Wh,b,next_h)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (D, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  x,Wx,prev_h,Wh,b,next_h = cache
  # dx, dprev_h, dWx, dWh, db = np.ones_like(x), np.ones_like(prev_h), np.ones_like(Wx), np.ones_like(Wh), np.ones_like(b) 
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  db = np.sum((1 - (next_h**2))*dnext_h,axis=0)

  dprev_h = ((1 - (next_h**2)) * dnext_h).dot(Wh.T) 
  dWh = prev_h.T.dot(((1 - (next_h**2)) * dnext_h))

  dx = ((1 - (next_h**2)) * dnext_h).dot(Wx.T)
  dWx = x.T.dot((1 - (next_h**2)) * dnext_h)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N,T,D = x.shape
  N,H = h0.shape
  h = np.zeros((N,T,H))
  next_h = np.zeros_like(h0) 
  cache = {}
  cache["major_info"] = (x,h0,Wx,Wh,b)
  for i in range(T):
    inp_x = x[:,i,:]
    if(i==0):
      prev_h = h0

    next_h, cache[i] = rnn_step_forward(inp_x.reshape(N,D), prev_h.reshape(N,H), Wx, Wh, b)
    h[:,i,:] += copy.deepcopy(next_h)
    prev_h = copy.deepcopy(next_h)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):

  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  x,h0,Wx,Wh,b = cache["major_info"]
  N,T,D = x.shape
  N,H = h0.shape
  dx = np.zeros_like(x)
  dh0 = np.zeros_like(h0)
  dWx = np.zeros_like(Wx)
  dWh = np.zeros_like(Wh)
  db = np.zeros_like(b)
  dprev_h_step = dh[:,T-1,:].reshape(N,H)
  # print dh[:,T-1,:]

  for i in reversed(range(T)):
    # print i#dh[:,i,:].reshape(N,H)
    dnext_h = dh[:, i, :] if i == (T-1) else dh[:, i, :] + dprev_h_step
    dx_step, dprev_h_step, dWx_step, dWh_step, db_step = rnn_step_backward(dnext_h, cache[i])
    # dx_step, dprev_h_step, dWx_step, dWh_step, db_step = rnn_step_backward(dh[:,i,:], cache[i])
    # if(i!=0):
      # print i
      # dh[:,(i-1),:] = dh[:,(i-1),:] + copy.deepcopy(dprev_h_step)
    dx[:,i,:] = copy.deepcopy(dx_step)
    dWh += copy.deepcopy(dWh_step)
    dWx += copy.deepcopy(dWx_step)
    db += copy.deepcopy(db_step)
  dh0 = copy.deepcopy(dprev_h_step)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  N,T = x.shape
  V,D = W.shape
  out = np.zeros((N,T,D))
  for i in range(N):
    for j in range(T):
      out[i,j,:] = W[x[i,j]] 
  cache = (out,x,W)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  out,x,W = cache
  N,T = x.shape
  V,D = W.shape
  dW = np.zeros((V,D))
  for i in range(N):
    for j in range(T):
      dW[x[i,j]]+= dout[i,j,:]

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  N,H = prev_h.shape
  a = x.dot(Wx) + prev_h.dot(Wh) + b
  ai = a[:,:H]
  af = a[:,H:(2*H)]
  ao = a[:,(2*H):(3*H)]
  ag = a[:,(3*H):]
  i = sigmoid(ai)
  f = sigmoid(af)
  o = sigmoid(ao)
  g = np.tanh(ag)
  next_c = (f * prev_c ) +  (i * g)
  next_h = (o) *  np.tanh(next_c)
  cache = (a,ai,af,ao,ag,i,f,o,g,next_c,next_h,prev_h,prev_c,Wx,Wh,x,b)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  a,ai,af,ao,ag,i,f,o,g,next_c,next_h,prev_h,prev_c,Wx,Wh,x,b = cache
  N,H = prev_h.shape

  dx = np.zeros_like(x)
  dWx = np.zeros_like(Wx)
  dWh = np.zeros_like(Wh)
  db = np.zeros_like(b)
  dprev_h = np.zeros_like(prev_h)
  dprev_c = np.zeros_like(prev_c)
  
  do_dao = o*(1-o)
  di_dai = i*(1-i)
  df_daf = f*(1-f)
  dg_dag = (1- (g**2))

  dai_dx = Wx[:,:(H)]
  daf_dx = Wx[:,H:(2*H)]
  dao_dx = Wx[:,(2*H):(3*H)]
  dag_dx = Wx[:,(3*H):(4*H)]

  dai_dWx = x
  daf_dWx = x
  dao_dWx = x
  dag_dWx = x

  dai_dWh = prev_h
  daf_dWh = prev_h
  dao_dWh = prev_h
  dag_dWh = prev_h

  dai_dprev_h = Wh[:,:H]
  daf_dprev_h = Wh[:,H:2*H]
  dao_dprev_h = Wh[:,2*H:3*H]
  dag_dprev_h = Wh[:,3*H:4*H]

  # Given : E = next_h = o dot tanh(next_c)
  # Given dE_dh as dnext_h
  # Computing dE_do = dnext_h * tanh(ct)
  # Computing dE_dnextc = dnext_h * (o * (1 - (next_c**2)))
  dE_dnextc = dnext_h * (o * (1 - (np.tanh(next_c)**2)))
  dE_do = dnext_h * np.tanh(next_c)

  dE_dx = (dE_do * do_dao).dot(dao_dx.T)
  dE_dWx = dao_dWx.T.dot(dE_do * do_dao)
  dE_dWh = dao_dWh.T.dot(dE_do * do_dao)
  dE_db = dE_do * do_dao
  dE_dprev_h = (dE_do * do_dao).dot(dao_dprev_h.T)

  dnext_c += dE_dnextc

# dWx = x.T.dot((1 - (next_h**2)) * dnext_h)
  # Given M = nextc = f dot prev_c + i dot g
  # Given dM_dnextc
  # computing dM_dprev_c = dnext_c * 
  dM_dprev_c = dnext_c * f
  dM_df = dnext_c * prev_c
  dM_di = dnext_c * g
  dM_dg = dnext_c * i

  dM_dx = (dM_df * df_daf).dot( daf_dx.T) + (dM_di * di_dai).dot( dai_dx.T) + (dM_dg * dg_dag).dot(dag_dx.T)
  dM_dprev_h = (dM_df * df_daf).dot(daf_dprev_h.T) + (dM_di * di_dai).dot(dai_dprev_h.T) + (dM_dg * dg_dag).dot(dag_dprev_h.T)

  dx = dM_dx+dE_dx

  # dM_dWx = dM_df * df_daf * daf_dWx + dM_di * di_dai * dai_dWx + dM_dg * dg_dag * dag_dWx
  dWx[:,:H] = dai_dWx.T.dot(dM_di * di_dai)
  dWx[:,H:2*H] = daf_dWx.T.dot(dM_df * df_daf)
  dWx[:,2*H:3*H] = dE_dWx
  dWx[:,3*H:] = dag_dWx.T.dot(dM_dg * dg_dag) 


  dprev_h = dM_dprev_h + dE_dprev_h
  
  # dM_dWh = daf_dWh.T.dot(dM_df * df_daf)  + dai_dWh.T.dot(dM_di * di_dai) + dag_dWh.T.dot(dM_dg * dg_dag) 
  dWh[:,:H] = dai_dWh.T.dot(dM_di * di_dai)
  dWh[:,H:2*H] = daf_dWh.T.dot(dM_df * df_daf)
  dWh[:,2*H:3*H] = dE_dWh
  dWh[:,3*H:] = dag_dWh.T.dot(dM_dg * dg_dag) 


  # dM_db = dM_df * df_daf + dM_di * di_dai + dM_dg * dg_dag 
  db[:H] = np.sum(dM_di * di_dai, axis=0)
  db[H:2*H] = np.sum(dM_df * df_daf, axis=0)
  db[2*H:3*H] = np.sum(dE_db, axis=0)
  db[3*H:] = np.sum(dM_dg * dg_dag, axis=0)


        # dnext_c += dnext_h * (o*(1-(next_c**2))*dnext_c  + next_c*(o*(1-o)))
        # dnext_c +=  (o*(1-(next_c**2)))

  dprev_c = dnext_c * f

        # do = o * (1-o)
        # di = i * (1-i)
        # df = f * (1-f)
        # dg = (1-(g**2))


        # COMPUTATION FOR DB
        # db = dnext_h*(o*(do) + dnext_c*(i*dg + g*di)) 


  # f*dprev_c
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N,T,D = x.shape
  N,H = h0.shape
  h = np.zeros((N,T,H))
  cache = {}
  cache["major_info"] = (x,h0,Wx,Wh,b)
  prev_c = np.zeros((N,H))
  for i in range(T):
    inp_x = x[:,i,:]
    if(i==0):
      prev_h = h0
    next_h, next_c, cache[i] = lstm_step_forward(inp_x.reshape(N,D), prev_h.reshape(N,H), prev_c, Wx, Wh, b)
    h[:,i,:] += copy.deepcopy(next_h)
    prev_h = copy.deepcopy(next_h)
    prev_c = copy.deepcopy(next_c)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  x,h0,Wx,Wh,b = cache["major_info"]
  N,T,D = x.shape
  N,H = h0.shape
  dx = np.zeros_like(x)
  dh0 = np.zeros_like(h0)
  dWx = np.zeros_like(Wx)
  dWh = np.zeros_like(Wh)
  db = np.zeros_like(b)
  dprev_c_step = np.zeros_like(h0)
  dprev_h_step = dh[:,T-1,:].reshape(N,H)
  # print dh[:,T-1,:]

  for i in reversed(range(T)):
    # print i#dh[:,i,:].reshape(N,H)
    o = cache[i][7]
    next_c = cache[i][9]
    # dnext_h = dh[:, i, :]
    # dE_dnextc = dnext_h * (o * (1 - (np.tanh(next_c)**2)))
    # dnext_c = dE_dnextc if i == (T-1) else (dE_dnextc + dprev_c_step)
    dnext_h = dh[:, i, :] if i == (T-1) else dh[:, i, :] + dprev_h_step
    # dE_dnextc = dnext_h * (o * (1 - (np.tanh(next_c)**2)))
    # dnext_c = dE_dnextc if i == (T-1) else (dE_dnextc + dprev_c_step)
    dx_step, dprev_h_step, dprev_c_step, dWx_step, dWh_step, db_step = lstm_step_backward(dnext_h, dprev_c_step, cache[i])
    dx[:,i,:] = copy.deepcopy(dx_step)
    dWh += copy.deepcopy(dWh_step)
    dWx += copy.deepcopy(dWx_step)
    db += copy.deepcopy(db_step)
  dh0 = copy.deepcopy(dprev_h_step)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  # print x.shape
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

