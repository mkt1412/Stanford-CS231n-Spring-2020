from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    N = x.shape[0]
    D = w.shape[0]
    M = w.shape[1]
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_r = x.reshape(N, -1)

    if x_r.shape != (N, D):
      raise Exception('x gets wrong dimension!')
      
    out = x_r.dot(w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    N = x.shape[0]
    D = w.shape[0]
    M = w.shape[1]
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(N, -1).T.dot(dout)
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # only positive x value has gradient flow
    mask = np.zeros(x.shape)
    mask[x >= 0] = 1
    dx = dout * mask

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.

    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation ofm) * sample_mean
          running_var = momentum * running_var + (1 - momentum) * sample_vare (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = np.zeros_like(x), None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        sample_mean = np.mean(x, axis=0)

        dev_from_mean = x - sample_mean

        dev_from_mean_sq = dev_from_mean ** 2

        sample_var = 1/N * np.sum(dev_from_mean_sq, axis=0)

        sample_stddev = np.sqrt(sample_var + eps)
        
        inverted_stddev = 1 / sample_stddev

        x_norm = dev_from_mean * inverted_stddev

        scaled_x = gamma * x_norm
        
        out = scaled_x + beta

        # store intermediate values in cache
        cache = {'mean': sample_mean, 'var': sample_var, 'stddev': sample_stddev, 
                          'beta': beta, 'gamma': gamma, 'x_norm': x_norm, 'x': x,
                          'eps': eps, 'dev_from_mean': dev_from_mean, 
                          'inverted_stddev': inverted_stddev, 'scaled_x': scaled_x} 
        
        # compte running mean and var
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var


        # navie version using iterations
        # for i in range(D):
        #   sample_mean = np.mean(x[:, i])
        #   sample_var = np.var(x[:, i])
        #   xi_norm = (x[:, i] - sample_mean) / np.sqrt(sample_var + eps)
        #   out[:, i] = gamma[i] * xi_norm + beta[i]

        #   running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        #   running_var = momentum * running_var + (1 - momentum) * sample_var

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D = dout.shape

    beta, gamma, x_norm, var, eps, stddev, dev_from_mean, inverted_stddev, x, mean = \
    cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
    cache['stddev'], cache['dev_from_mean'], cache['inverted_stddev'], cache['x'], \
    cache['mean']
 
    dbeta = np.sum(dout, axis=0)
    dscaled_x = dout

    dgamma = np.sum(x_norm * dscaled_x, axis=0)
    dx_norm = gamma * dscaled_x

    dinverted_stddev = np.sum(dev_from_mean * dx_norm, axis=0)
    ddev_from_mean = inverted_stddev * dx_norm

    dstddev = -1/(stddev**2) * dinverted_stddev

    dvar = (0.5) * 1/np.sqrt(var + eps) * dstddev

    ddev_from_mean_sq = 1/N * np.ones((N,D)) * dvar 

    ddev_from_mean += 2 * dev_from_mean * ddev_from_mean_sq

    dx = ddev_from_mean
    dmean = -1 * np.sum(ddev_from_mean, axis=0)

    dx += 1./N * np.ones((N,D)) * dmean

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    beta, gamma, x_norm, var, eps, stddev, dev_from_mean, inverted_stddev, mean, x = \
    cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
    cache['stddev'], cache['dev_from_mean'], cache['inverted_stddev'], cache['mean'], \
    cache['x']

    N = dout.shape[0] 

    dbeta = np.sum(dout, axis=0)
    dscaled_x = dout 

    dgamma = np.sum((x - mean) * (var + eps)**(-1. / 2.) * dout, axis=0)

    dmean = 1/N * np.sum(dout, axis=0)
    dvar = 2/N * np.sum(dev_from_mean * dout, axis=0)
    dstddev = dvar/(2 * stddev)
    dx = gamma*((dout - dmean)*stddev - dstddev*(dev_from_mean))/stddev**2   

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x = x.T # (D, N)
    D, N = x.shape

    sample_mean = np.mean(x, axis=0) # (N,)

    dev_from_mean = x - sample_mean # (D, N)

    dev_from_mean_sq = dev_from_mean ** 2 # (D, N)

    sample_var = 1/D * np.sum(dev_from_mean_sq, axis=0) # sum over D

    sample_stddev = np.sqrt(sample_var + eps) # (N,)
    
    inverted_stddev = 1 / sample_stddev # (N,)

    x_norm = dev_from_mean * inverted_stddev# (D, N)

    scaled_x = gamma * x_norm.T # (N, D)
    
    out = scaled_x + beta # (N, D)

    # store intermediate values in cache
    cache = {'mean': sample_mean, 'var': sample_var, 'stddev': sample_stddev, 
                      'beta': beta, 'gamma': gamma, 'x_norm': x_norm, 'x': x,
                      'eps': eps, 'dev_from_mean': dev_from_mean, 
                      'inverted_stddev': inverted_stddev, 'scaled_x': scaled_x} 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = dout.shape

    beta, gamma, x_norm, var, eps, stddev, dev_from_mean, inverted_stddev, x, mean = \
    cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
    cache['stddev'], cache['dev_from_mean'], cache['inverted_stddev'], cache['x'], \
    cache['mean']
 
    dbeta = np.sum(dout, axis=0) # (D,)
    dscaled_x = dout # (N, D)

    dgamma = np.sum(x_norm.T * dscaled_x, axis=0) # (D,)
    dx_norm = gamma * dscaled_x # (N ,D)

    dinverted_stddev = np.sum(dev_from_mean * dx_norm.T, axis=0) # (N,)
    ddev_from_mean = inverted_stddev * dx_norm.T # (N, D)

    dstddev = -1/(stddev**2) * dinverted_stddev # (N,)

    dvar = (0.5) * 1/np.sqrt(var + eps) * dstddev # (N,)

    ddev_from_mean_sq = 1/D * np.ones((D,N)) * dvar # (D, N)

    ddev_from_mean += 2 * dev_from_mean * ddev_from_mean_sq # (D, N)

    dx = ddev_from_mean # (D, N)
    dmean = -1 * np.sum(ddev_from_mean, axis=0) # (N,)

    dx += 1/D * np.ones((D,N)) * dmean # (D, N)

    dx = dx.T # (N, D)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = np.random.binomial(1, p, x.shape)
        out = x * mask / p


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]
    p = dropout_param['p']

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask / p

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, k, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - k: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    stride = conv_param['stride']
    pad = conv_param['pad']
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N0, C0, H0, W0 = x.shape
    
    # padding horizontally
    x_padded = x
    h_padding = np.zeros((N0, C0, H0, 1))
    x_padded = np.concatenate((h_padding, x, h_padding), axis=3)
    
    # padding vertically
    N1, C1, H1, W1 = x_padded.shape
    v_padding = np.zeros((N1, C1, 1, W1))
    x_padded = np.concatenate((v_padding, x_padded, v_padding), axis=2)

    # double check
    N2, C2, H2, W2 = x_padded.shape

    if (H2 != H0 + 2 * pad) or (W2 != W0 + 2 * pad):
      raise Exception('Wrong dimension after zero padding!')
      
    # Convolution
    F, C, HH, WW = k.shape

    # shape of output feature map
    H_out = int(1 + (H2 - HH) / stride)
    W_out = int(1 + (W2 - WW) / stride)
    out = np.zeros((N2, F, H_out, W_out))

    # Iteration through images
    for n in range(N2):
      img = x_padded[n]
      for f in range(F):
        kernel = k[f]
        bias = b[f]

        i = 0 
        for h in range(0, H2, stride):
          # print("h: " + str(h) + " HH: " + str(HH) + " H2: " + str(H2))
          j = 0
          if h + HH > H2:
            break
          for w in range(0, W2, stride):
            # print("w: " + str(w) + " WW: " + str(WW) + " W2: " + str(W2))
            if w + WW > W2:
              break
            img_patch = img[:, h:h+HH, w:w+WW]
            out[n, f, i, j] = (kernel * img_patch).sum() + bias

            j += 1
          i += 1
    
    cache = (x, k, b, conv_param)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # x: (N, C, H, W)
    # k: (F, C, HH, WW)
    # b: (F,)
    x, k, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']

    # restore x to x_padded
    N0, C0, H0, W0 = x.shape
    x_padded = x
    h_padding = np.zeros((N0, C0, H0, 1))
    x_padded = np.concatenate((h_padding, x, h_padding), axis=3)
    N1, C1, H1, W1 = x_padded.shape
    v_padding = np.zeros((N1, C1, 1, W1))
    x_padded = np.concatenate((v_padding, x_padded, v_padding), axis=2)

    # subtract params
    N2, C2, H2, W2 = x_padded.shape
    N2, F, H_out, W_out = dout.shape
    F, C, HH, WW = k.shape
    dx_padded = np.zeros(x_padded.shape)
    dk = np.zeros(k.shape)
    db = np.zeros(b.shape)

    # Iteration through images
    for n in range(N2):
      img = x_padded[n]
      for f in range(F):
        kernel = k[f]

        i = 0 
        for h in range(0, H2, stride):
          j = 0
          if h + HH > H2:
            break
          for w in range(0, W2, stride):
            if w + WW > W2:
              break
            img_patch = img[:, h:h+HH, w:w+WW]
            db[f] += dout[n, f, i, j]
            dx_padded[n, :, h:h+HH, w:w+WW] += dout[n, f, i, j] * kernel
            dk[f] += dout[n, f, i, j] * img_patch

            j += 1
          i += 1
    

    # crop dx_padded to dx
    dx = dx_padded[:, :, pad:H2-pad, pad:W2-pad]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dk, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # get params
    N, C, H, W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']

    # shape of output feature map
    H_out = int(1 + (H - HH) / stride)
    W_out = int(1 + (W - WW) / stride)
    out = np.zeros((N, C, H_out, W_out))

    # Iteration through images
    for n in range(N):
      for c in range(C):
        img = x[n, c]

        i = 0 
        for h in range(0, H, stride):
          # print("h: " + str(h) + " HH: " + str(HH) + " H2: " + str(H2))
          j = 0
          if h + HH > H:
            break
          for w in range(0, W, stride):
            # print("w: " + str(w) + " WW: " + str(WW) + " W2: " + str(W2))
            if w + WW > W:
              break
            img_patch = img[h:h+HH, w:w+WW]
            out[n, c, i, j] = np.max(img_patch)

            j += 1
          i += 1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # get params
    x, pool_param = cache
    N, C, H, W = x.shape
    dx = np.zeros(x.shape)
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']

    # Iteration through images
    for n in range(N):
      for c in range(C):
        img = x[n, c]

        i = 0 
        for h in range(0, H, stride):
          # print("h: " + str(h) + " HH: " + str(HH) + " H2: " + str(H2))
          j = 0
          if h + HH > H:
            break
          for w in range(0, W, stride):
            # print("w: " + str(w) + " WW: " + str(WW) + " W2: " + str(W2))
            if w + WW > W:
              break
            img_patch = img[h:h+HH, w:w+WW]
            max_ind = np.unravel_index(img_patch.argmax(), img_patch.shape)
            dx[n, c, h + max_ind[0], w + max_ind[1]] = dout[n, c, i, j]

            j += 1
          i += 1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    dx, dgamma, dbeta = batchnorm_backward(dout, cache)

    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = np.zeros_like(x), None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    size = (N*G, C//G*H*W) # (N1, D1)

    x = x.reshape((N*G, -1)) # (N1, D1)

    mean = np.mean(x, axis=1, keepdims=True) # (N1, 1)
    """
    (X, Y) - (Y,)    CORRECT
    (X, Y) - (X,)     WRONG
    (X, Y) - (X, 1)  CORRECT
    """

    dev_from_mean = x - mean # (N1, D1)

    dev_from_mean_sq = dev_from_mean ** 2 # (N1, D1)

    var = 1 / size[1] * np.sum(dev_from_mean_sq, axis=1, keepdims=True) # (N1, 1)

    stddev = np.sqrt(var + eps) # (N1, 1)

    inverted_stddev = 1 / stddev # (N1, 1)

    x_norm = dev_from_mean * inverted_stddev # (N1, D1)
    x_norm = x_norm.reshape(N, C, H, W)

    scaled_x = gamma * x_norm # (N1, D1)

    out = scaled_x + beta # (N1, D1)

    # store intermediate values in cache
    cache = {'mean': mean, 'stddev': stddev, 'var': var, 'gamma': gamma, \
             'beta': beta, 'eps': eps, 'x_norm': x_norm, 'dev_from_mean': dev_from_mean, \
             'inverted_stddev': inverted_stddev, 'x': x, 'size': size, 'G': G, 'scaled_x': scaled_x}


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    beta, gamma, x_norm, var, eps, stddev, dev_from_mean, inverted_stddev, x, mean, size, G, scaled_x = \
    cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
    cache['stddev'], cache['dev_from_mean'], cache['inverted_stddev'], cache['x'], \
    cache['mean'], cache['size'], cache['G'], cache['scaled_x']

    N, C, H, W = dout.shape
    
    dbeta = np.sum(dout, axis = (0,2,3), keepdims = True) #(1, C, 1, 1)
    dscaled_x = dout # (N, C, H, W)

    dgamma = np.sum(dscaled_x * x_norm, axis = (0,2,3), keepdims = True) #(1, C, 1, 1)
    dx_norm = dscaled_x * gamma # (N, C, H, W)
    dx_norm = dx_norm.reshape(size) # (N1, D1)

    dinverted_stddev = np.sum(dx_norm * dev_from_mean, axis = 1, keepdims = True) # (N1, 1)
    ddev_from_mean = dx_norm * inverted_stddev # (N1, D1)

    dstddev = (-1/(stddev**2)) * dinverted_stddev # (N1, 1)

    dvar = 0.5 * (1/np.sqrt(var + eps)) * dstddev # (N1, 1)

    ddev_from_mean_sq = (1/size[1]) * np.ones(size) * dvar # NxD = NxD*N
   
    ddev_from_mean += 2 * dev_from_mean * ddev_from_mean_sq # (N1, D1)

    dx = (1) * ddev_from_mean # (N1, D1)
    dmean = -1 * np.sum(ddev_from_mean, axis = 1, keepdims = True) # (N1, 1)

    dx += (1/size[1]) * np.ones(size) * dmean # (N1, D1)

    dx = dx.reshape(N, C, H, W) # (N, C, H, W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
