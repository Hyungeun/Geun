import tensorflow as tf


def Conv2D(input, kernel_shape, strides, padding, name='Conv2d'):
    '''
    Convolutional layer.
    '''
    with tf.variable_scope(name):
        W = tf.get_variable("W", kernel_shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable("b", (kernel_shape[-1]),initializer=tf.constant_initializer(value=0.0))
    return tf.nn.conv2d(input, W, strides, padding) + b


def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add\
                      (tf.nn.conv2d(img, w,\
                                    strides=[1,1,1,1],\
                                    padding='SAME'),b))

def max_pool(img, k_row, k_col):
    return tf.nn.max_pool(img, \
                          ksize=[1, k_row, k_col, 1],\
                          strides=[1, k_row, k_col, 1],\
                          padding='VALID')

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)   
        
def batchnorm(input, is_train, decay=0.999, name='BatchNorm'):
    '''
    https://github.com/zsdonghao/tensorlayer/blob/master/tensorlayer/layers.py
    https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
    http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow
    '''
    from tensorflow.python.training import moving_averages
    from tensorflow.python.ops import control_flow_ops
    
    axis = list(range(len(input.get_shape()) - 1))
    fdim = input.get_shape()[-1:]
    
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', fdim, initializer=tf.constant_initializer(value=0.0))
        gamma = tf.get_variable('gamma', fdim, initializer=tf.constant_initializer(value=1.0))
        moving_mean = tf.get_variable('moving_mean', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)
        moving_variance = tf.get_variable('moving_variance', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)
        
  
        def mean_var_with_update():
            batch_mean, batch_variance = tf.nn.moments(input, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=True)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_variance, decay, zero_debias=True)
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        mean, variance = control_flow_ops.cond(is_train, mean_var_with_update, lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(input, mean, variance, beta, gamma, 1e-3)
    
def Dense(input, weight_shape, name='Dense'):
    '''
    Fully connected layer. 
    '''
    with tf.variable_scope(name):
        W = tf.get_variable("W", weight_shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable("b", (weight_shape[-1]),initializer=tf.constant_initializer(value=0.0))
    return tf.matmul(input, W) + b
