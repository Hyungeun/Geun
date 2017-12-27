import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten

from data_function import data_preprocessor_sogang, batch_provider_new
from nn_function import lrelu

#%% load data
x_train, y_train, x_test, y_test, train_max, train_min = data_preprocessor_sogang('sogang_component_proc.csv')

#%%
n_day = 353 
n_home= 20
n_train = 300
n_test = 53
n_sample = n_day*n_home
n_train_sample = n_train*n_home
n_test_sample = n_test*n_home
ts=96
 
#%%
growth_rate = 30
num_blocks = 2  # number of dense block + transition layers
num_layers_1 = 8
num_layers_last = 8

max_pool_size = [3,1]
max_pool_stride = [1,1]
avg_pool_size = [3,1]
avg_pool_stride = [1,1]
conv_weight = [2,4]
conv0_weight = [3,3]
conv0_stride = [1,1]

init_learning_rate = 1e-1
dropout_rate = 0.2
    
momentum = 0.9
weight_decay = 1e-4
    
num_classes = 96
batch_size = 100
epochs = 65

#%%
def BN(x, training, scope):
    with arg_scope(
        [batch_norm],
        scope=scope,
        updates_collections=None,
        decay=0.9,
        center=True,
        scale=True,
        zero_debias_moving_mean=True):
        return tf.cond(training,
            lambda: batch_norm(inputs=x, is_training=training, reuse=None),
            lambda: batch_norm(inputs=x, is_training=training, reuse=True))
    
def ReLU(x): 
    return tf.nn.relu(x)
    
def Conv(input, filter, kernel, stride=1, layer_name='conv'):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(
            inputs=input,
            filters=filter,
            kernel_size=kernel,
            strides=stride,
            padding='SAME')
        return network
    
def Dropout(x, rate, training): 
    return tf.layers.dropout(inputs=x, training=training, rate=rate)
    
def Avg_Pool(x, pool_size=avg_pool_size, stride=avg_pool_stride, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
    
def Global_Avg_Pool(x, stride=1):
    pool_size = np.shape(x)[1:3]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)
    
def Max_Pool(x, pool_size=max_pool_size, stride=max_pool_stride, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
    
def Linear(x): 
    return tf.layers.dense(inputs=x, units=num_classes, name='linear')
    
def Concatenation(layers): 
    return tf.concat(layers, axis=3)
           
#%% densenet model 
##################################################################################################
class DenseNet(object):
    def __init__(self, x, num_blocks, filters, filters_last, training):
            
        self.num_blocks = num_blocks
        self.filters = filters
        self.filters_last = filters_last
        self.training = training
            
        self.model = self.create_model(x)
        
    def create_model(self, x):
        
        print('0',x.shape)
        x = Conv(x, filter=2 * self.filters, kernel=conv0_weight, stride=conv0_stride, layer_name='conv0')
        print('1',x.shape)
        x = Max_Pool(x)
        print('2',x.shape)
            
        for i in range(self.num_blocks):
            x = self.dense_block(x, num_layers=num_layers_1, layer_name='dense_%d' % i)
            print('3',x.shape)
            x = self.transition(x, scope='trans_%d' % i)
            print('4',x.shape)
                
        x = self.dense_block_last(x, num_layers=num_layers_last, layer_name='dense_final')
        print('5',x.shape)         
        x = BN(x, training=self.training, scope='linear_batch')
        x = lrelu(x)

        x=tf.reshape(x,shape=[-1,ts]) 
        print('6',x.shape)
        
        y_final_prime2 = x
        y_final_ = tf.slice(x,[0,0],[-1,95])
        y_final_prime1 = tf.concat([x_init,y_final_],1)

        w1_final = tf.Variable(tf.random_normal([96,96])) 
        y_pred =tf.add(tf.matmul(y_final_prime1, w1_final),y_final_prime2)
        x = lrelu(y_pred)
        return x
        
    def bottleneck(self, x, scope):
        with tf.name_scope(scope):
            x = BN(x, training=self.training, scope=scope + '_batch1')
            x = lrelu(x)
            x = Conv(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
#            x = Dropout(x, rate=dropout_rate, training=self.training)
                
            x = BN(x, training=self.training, scope=scope + '_batch2')
            x = lrelu(x)
            x = Conv(x, filter=self.filters, kernel=conv_weight, layer_name=scope + '_conv2')
#            x = Dropout(x, rate=dropout_rate, training=self.training)        
            
            return x
            
    def bottleneck_last(self, x, scope):
        with tf.name_scope(scope):
            x = BN(x, training=self.training, scope=scope + '_batch1_last')
            x = lrelu(x)
            x = Conv(x, filter=4 * self.filters_last, kernel=[1, 1], layer_name=scope + '_conv1_last')
#            x = Dropout(x, rate=dropout_rate, training=self.training)
                
            x = BN(x, training=self.training, scope=scope + '_batch2')
            x = lrelu(x)
            x = Conv(x, filter=self.filters_last, kernel=conv_weight, layer_name=scope + '_conv2_last')
#            x = Dropout(x, rate=dropout_rate, training=self.training)        
            
            return x            
           
    def transition(self, x, scope):
        with tf.name_scope(scope):
            x = BN(x, training=self.training, scope=scope + '_batch1')
            x = lrelu(x)
            x = Conv(x, filter=self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
#            x = Dropout(x, rate=dropout_rate, training=self.training)
            x = Avg_Pool(x)
            
            return x
        
    def dense_block(self, x, num_layers, layer_name):
        with tf.name_scope(layer_name):
            concat_layers = [x]
                
            bc = self.bottleneck(x, scope=layer_name + '_bottleneck_0')
            concat_layers.append(bc)
                
            for i in range(num_layers - 1):
                y = Concatenation(concat_layers)
                bc = self.bottleneck(y, scope=layer_name + '_bottleneck_%d' % (i + 1))
                concat_layers.append(bc)
             
            return bc
            
    def dense_block_last(self, x, num_layers, layer_name):
        with tf.name_scope(layer_name):
            concat_layers = [x]
                
            bc = self.bottleneck_last(x, scope=layer_name + '_bottleneck_last_0')
            concat_layers.append(bc)
                
            for i in range(num_layers - 1):
                y = Concatenation(concat_layers)
                bc = self.bottleneck_last(y, scope=layer_name + '_bottleneck_last_%d' % (i + 1))
                concat_layers.append(bc)
             
            return bc
##################################################################################################             
#%% network input
x = tf.placeholder(tf.float32, shape=[None, ts*7],name='x')
label = tf.placeholder(tf.float32, shape=[None, ts],name='y')
x_init = tf.placeholder(tf.float32, [None, 1]) 
training_flag = tf.placeholder(tf.bool, name='training_flg')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
batch_images = tf.reshape(x, [-1,7,ts,1])
    
logits = DenseNet(x=batch_images, num_blocks=num_blocks, filters=growth_rate, filters_last=1, training=training_flag).model
    
cost = tf.reduce_mean(tf.square(label-logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)
    
#%% session start
avg_loss=0
avg_loss_val=0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    saver = tf.train.Saver()
    save_path = "densenet_sogang_t/model.ckpt"
    
    epoch_learning_rate = init_learning_rate
        
    for epoch in range(epochs):
        if epoch in [epochs // 2, (epochs * 3) // 4]: 
            epoch_learning_rate //= 10
        
        for batch in range(n_train_sample // batch_size):
            batch_x, batch_y, batch_x_init = batch_provider_new(x_train, y_train, num_data=n_train_sample, batch_size=batch_size)
    
            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                x_init: batch_x_init,
                learning_rate: epoch_learning_rate,
                training_flag : True}
    
            _, train_loss = sess.run([train, cost], feed_dict=train_feed_dict)
    
            if batch % 10 == 0:                   
                print( 'step:',batch, 'loss:',train_loss)
                
            batch_xt, batch_yt, batch_xit = batch_provider_new(x_test, y_test, num_data=n_test_sample, batch_size=batch_size)
   
            test_feed_dict = {
                x: batch_xt,
                label: batch_yt,
                x_init: batch_xit,
                learning_rate: epoch_learning_rate,
                training_flag : False}

            test_loss = sess.run(cost, feed_dict=test_feed_dict)
            
            
            avg_loss +=train_loss            
            avg_loss_val +=test_loss
                                
        avg_loss_val = avg_loss_val/(n_train_sample//batch_size)
        avg_loss = avg_loss/(n_train_sample//batch_size)
        
        print("Epoch:", '%d' %(epoch+1), "loss_avg_train",avg_loss)
        print("Epoch:", '%d' %(epoch+1), 'loss_avg_val',avg_loss_val)
                      
    print("Optimization Finished!")
    saver.save(sess, save_path)

