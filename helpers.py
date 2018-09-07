import numpy as np
import tensorflow as tf

###############################################################################
def flatten(x, name=None, reuse=None):
    """Flatten Tensor to 2-dimensions.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to flatten.
    name : None, optional
        Variable scope for flatten operations

    Returns
    -------
    flattened : tf.Tensor
        Flattened tensor.
    """
    with tf.variable_scope('flatten'):
        dims = x.get_shape().as_list()
        if len(dims) == 4:
            flattened = tf.reshape(
                x,
                shape=[-1, dims[1] * dims[2] * dims[3]])
        elif len(dims) == 2 or len(dims) == 1:
            flattened = x
        else:
            raise ValueError('Expected n dimensions of 1, 2 or 4.  Found:',
                             len(dims))

        return flattened


###############################################################################
def linear(X, n_outputs, scope=None, activation=None, reuse=None):
    if X.get_shape() != 2:
        X = flatten(X)
        
    n_inputs = X.get_shape().as_list()[1]
    
    with tf.variable_scope(scope or "fully_connected", reuse=reuse):

        weights = tf.get_variable(
            name="weights",
            dtype=tf.float32,
            shape=[n_inputs, n_outputs],
            initializer=tf.random_normal_initializer(stddev=0.1)
        )

        biases = tf.get_variable(
            name="biases",
            dtype=tf.float32,
            shape=[n_outputs],
            initializer=tf.constant_initializer(value=0)
        )

        hidden = tf.matmul(X, weights) + biases
        
        if activation:
            hidden = activation(hidden)
            
        return hidden

###############################################################################
def im_split(img):
    xs = []     # To store positions (x, y)
    ys = []     # to store Colors (R, G, B)
    
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            xs.append([row_i, col_i])
            ys.append(img[row_i, col_i])

    xs = np.array(xs)
    ys = np.array(ys)

    return xs, ys
