import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm,flatten

from data_function import data_preprocessor_sogang, batch_provider_new, to_csv_file, init_func
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
growth_rate = 10
num_blocks = 2  # number of dense block + transition layers
num_layers_1 = 8
num_layers_last = 8

max_pool_size = [3,1]
max_pool_stride = [1,1]
avg_pool_size = [3,1]
avg_pool_stride = [1,1]
conv_weight = [2,3]
conv0_weight = [3,3]
conv0_stride = [1,1]

init_learning_rate = 1e-1
dropout_rate = 0.2
    
momentum = 0.9
weight_decay = 1e-4
    
num_classes = 96
batch_size = 100
epochs = 100

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
training_flag = tf.placeholder(tf.bool, name='training_flg')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
batch_images = tf.reshape(x, [-1,7,ts,1])
    
logits = DenseNet(x=batch_images, num_blocks=num_blocks, filters=growth_rate, filters_last=1, training=training_flag).model
    
cost = tf.reduce_mean(tf.square(label-logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

#%%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    saver = tf.train.Saver()
    save_path = "densenet_sogang_4/model.ckpt"
    saver.restore(sess, save_path)

    print('model restored !!')
    epoch_learning_rate = init_learning_rate

#%% test mape
    MAPE_matrix = []
    MAPE_max_matrix = []
    mape_perhome_matrix=[]
    for i in range(1) :
#        x_init_test = init_func(x_test[53*i:53*(i+1),:],53,53)
        y_predict_ = sess.run(logits, feed_dict={x: x_test[53*i:53*(i+1),:], label: y_test[53*i:53*(i+1),:],learning_rate: epoch_learning_rate,training_flag : False})         
        y_predict = np.array(y_predict_)
                                                     
        y_predict_denorm = y_predict*(train_max-train_min)+train_min
        y_test_denorm = y_test[53*i:53*(i+1),:]*(train_max-train_min)+train_min

        pred = y_predict_denorm
        true = y_test_denorm
        mape = [np.sum(abs(pred[j]-true[j])/true[j]*100)/ts for j in range(53)]
        
        n=0
        matrix_b=[]
        for a,b in enumerate(mape):
            if b < 100.0:
                aa = b
                matrix_b.append(aa)
                n+=1
        print('n',n)
        
        mape_perhome = np.mean(matrix_b)
        print('%d'%i,mape_perhome)
        mape_perhome_matrix.append(mape_perhome)
                
    print('mape',np.mean(mape_perhome_matrix))
    to_csv_file(y_predict_denorm, 'pred.csv',index=False)
    to_csv_file(y_test_denorm,'true.csv',index=False)
    y_predict_denorm = np.reshape(y_predict_denorm,[1,-1])
    y_test_denorm = np.reshape(y_test_denorm,[1,-1])
    to_csv_file(y_predict_denorm, 'pred_s.csv',index=False)
    to_csv_file(y_test_denorm, 'true_s.csv',index=False)
   
    MAPE_matrix = []
    MAPE_max_matrix = []
    mape_perhome_matrix=[]
    for i in range(1) :
#        x_init_train = init_func(x_train[300*i:300*(i+1),:],300,300)
        y_predict_ = sess.run(logits, feed_dict={x: x_train[300*i:300*(i+1),:], label: y_train[300*i:300*(i+1),:],learning_rate: epoch_learning_rate,training_flag : False})         
        y_predict = np.array(y_predict_)
                                                     
        y_predict_denorm = y_predict*(train_max-train_min)+train_min
        y_test_denorm = y_train[300*i:300*(i+1),:]*(train_max-train_min)+train_min
        
        pred = y_predict_denorm
        true = y_test_denorm
        mape = [np.sum(abs(pred[j]-true[j])/true[j]*100)/ts for j in range(53)]
        
        n=0
        matrix_b=[]
        for a,b in enumerate(mape):
            if b < 100.0:
                aa = b
                matrix_b.append(aa)
                n+=1
        print('n',n)
        
        mape_perhome = np.mean(matrix_b)
        print('%d'%i,mape_perhome)
        mape_perhome_matrix.append(mape_perhome)
                
    print('mape_train',np.mean(mape_perhome_matrix))
    to_csv_file(y_predict_denorm, 'pred_train.csv',index=False)
    to_csv_file(y_test_denorm,'true_train.csv',index=False)
    y_predict_denorm = np.reshape(y_predict_denorm,[1,-1])
    y_test_denorm = np.reshape(y_test_denorm,[1,-1])
    to_csv_file(y_predict_denorm, 'pred_s_train.csv',index=False)
    to_csv_file(y_test_denorm, 'true_s_train.csv',index=False)
