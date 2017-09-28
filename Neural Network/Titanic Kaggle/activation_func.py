import numpy as np

def sigmoid(Z):
    """
            Implement the Sigmoid function.
               1 
            --------
            1 + e^-z

    Arguments:
    Z      =   Output of any linear layer of any shape
    

    Outputs:
    A      =   Post-Activation parameter, of the same shape of Z
    cache  =   return Z, useful for back propagation

    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A,cache

def relu(Z):
    """
            Implement the RELU or Rectified Linear Unit function.
            max(0,Z)
            
    Arguments:
    Z       =    Output of the linear layer, of any shape

    Returns:
    A       =    Post-activation parameter, of the same shape as Z
    cache   =    a python dictionary containing "A" \n stored for computing the backward pass efficiently
    """
    A = np.maximum(0,Z)
    cache = Z
    assert(A.shape == Z.shape)
    return A, cache

def lrelu(Z):
    """
            Implement the Leaky RELU or Leaky Rectified Linear Unit function.
            max(0.01 * Z , Z)
            
    Arguments:
    Z       =    Output of the linear layer, of any shape

    Returns:
    A       =    Post-activation parameter, of the same shape as Z
    cache   =    a python dictionary containing "A" \n stored for computing the backward pass efficiently
    """
    cache = Z
    A = np.maximum(0.01*Z,Z)
    return A, cache
 

def tanh(Z):
    """
            Implement the TanH or Hyperbolic Tangent function.
            e^z - e^-z
            ----------
            e^z + e^-z
            
    Arguments:
    Z       =    Output of the linear layer, of any shape

    Returns:
    A       =    Post-activation parameter, of the same shape as Z
    cache   =    a python dictionary containing "A" \n stored for computing the backward pass efficiently
    """
    A = np.tanh(Z)
    cache = Z
    return A, cache


def back_sigmoid(dA, cache):
    assert(dA.shape[0] == 1)
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * (s * (1-s))
    
    assert(dZ.shape == Z.shape)
    return dZ

def back_relu(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    return dZ

def back_lrelu(dA, cache):
    pass

def back_tanh(dA, cache):
    pass



