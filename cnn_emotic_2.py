# CNN model for EMOTIC.

import tensorflow as tf
import numpy as np
from params import BN_EPS, STD_VAR_INI
import tools
import collections




###########################################################################################################
### Add one block of the form conv + relu + conv + bn + relu
def add_block(x_in, var_dict, strideH, strideW, padH, padW, path_id, block_id):
    # Base name, including path id and block id:
    basename = 'p' + str(path_id) + '_b' + str(block_id) + '_'
    # Block operations:
    conv1 = conv2d(x_in, var_dict.items()[0][1], var_dict.items()[1][1], strideH, strideW, padH, padW, name=basename+'conv1')
    relu1 = tf.nn.relu(conv1, name=basename+'relu1')
    conv2 = conv2d(relu1, var_dict.items()[2][1], var_dict.items()[3][1], strideW, strideH, padW, padH, name=basename+'conv2')
    bn = tf.nn.batch_normalization(conv2, var_dict.items()[4][1], var_dict.items()[5][1], \
        var_dict.items()[6][1], var_dict.items()[7][1], BN_EPS, name=basename+'bn')
    relu2 = tf.nn.relu(bn, name=basename+'relu2')
    return relu2


###########################################################################################################
### ********
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = STD_VAR_INI)
    return tf.Variable(initial, name=name)


###########################################################################################################
### Xavier uniform initialization
def weight_variable_xavier(shape, dimin, dimout, name):
    semilength = np.sqrt(2.0 / (dimin + dimout)) * np.sqrt(3)
    initial = tf.random_uniform(shape, minval = -semilength, maxval = semilength)
    return tf.Variable(initial, name=name)


###########################################################################################################
### ********
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name=name)


###########################################################################################################
### Convolution with bias, and custom stride and pad.
def conv2d(x, W, b, strideH=1, strideW=1, padH=0, padW=0, name=None):
    paddings = tf.constant([[0, 0], [padH, padH], [padW, padW], [0, 0]])
    return tf.add(tf.nn.conv2d(tf.pad(x, paddings), W, strides = [1, strideH, strideW, 1], padding = "VALID"), b, name=name)


########################################################################################
#### Main class
class cnn_builder_class:
    
    
    ###########################################################################################################
    ### *****
    def define_bodypath(self):
        
#         conv1_W = weight_variable([5, 5, 3, 32], 'conv1_W')
#         conv1_b = bias_variable([32], 'conv1_b')
#         b1_bn_mean, b1_bn_variance, b1_bn_offset, b1_bn_scale = self.bn_variables(32, 'b1_')
        
#         conv_new1_W = weight_variable([5, 5, 32, 32], 'conv_new1_W')
#         conv_new1_b = bias_variable([32], 'conv_new1_b')
#         conv_new2_W = weight_variable([5, 5, 64, 64], 'conv_new2_W')
#         conv_new2_b = bias_variable([64], 'conv_new2_b')
#         conv_new3_W = weight_variable([5, 5, 64, 64], 'conv_new3_W')
#         conv_new3_b = bias_variable([64], 'conv_new3_b')
        
#         conv2_W = weight_variable([5, 5, 32, 64], 'conv2_W')
#         conv2_b = bias_variable([64], 'conv2_b')
#         b2_bn_mean, b2_bn_variance, b2_bn_offset, b2_bn_scale = self.bn_variables(64, 'b2_')
        
        dense1_weight = weight_variable([128, 256], name='dense1_weight')
        dense1_bias = bias_variable([256], name='dense1_bias')
        dense2_weight = weight_variable([256, 1000], name='dense2_weight')
        dense2_bias = bias_variable([1000], name='dense2_bias')       
        
        x_in = tf.placeholder(tf.float32, shape=[None, 128, 128, 3], name='x_in')
        
#         x = conv2d(x_in, conv1_W, conv1_b, 1, 1, 2, 2, name='conv1')
#         x = tf.nn.relu(x, name='relu1')

#         var_dict = collections.OrderedDict()
#         self.random_variables_block(3, 16, 32, 5, 1, var_dict, 0, 0)
#         x = add_block(x_in, var_dict, 1, 1, 2, 0, 0, 0)
        
#         x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='maxpool1')
        
#         x = conv2d(x, conv_new1_W, conv_new1_b, 2, 2, 2, 2, name='conv_new1')
#         x = tf.nn.relu(x, name='relu_new1')

#         var_dict = collections.OrderedDict()
#         self.random_variables_block(32, 32, 32, 5, 1, var_dict, 10, 10)
#         x = add_block(x, var_dict, 2, 1, 2, 0, 10, 10)

        var_dict = collections.OrderedDict()
        self.random_variables_block(3, 32, 64, 3, 1, var_dict, 10, 10)
        x = add_block(x_in, var_dict, 2, 1, 1, 0, 10, 10)
        
#         x = tf.nn.batch_normalization(x, b1_bn_mean, b1_bn_variance, b1_bn_offset, b1_bn_scale, BN_EPS, name='b1_bn')
#         x = conv2d(x, conv2_W, conv2_b, 1, 1, 2, 2, name='conv2')
#         x = tf.nn.relu(x, name='relu2')

#         var_dict = collections.OrderedDict()
#         self.random_variables_block(32, 48, 64, 5, 1, var_dict, 15, 15)
#         x = add_block(x, var_dict, 1, 1, 2, 0, 15, 15)
        
#         x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='maxpool2')
        
#         x = conv2d(x, conv_new2_W, conv_new2_b, 2, 2, 2, 2, name='conv_new2')
#         x = tf.nn.relu(x, name='relu_new2')

#         var_dict = collections.OrderedDict()
#         self.random_variables_block(64, 64, 64, 5, 1, var_dict, 20, 20)
#         x = add_block(x, var_dict, 2, 1, 2, 0, 20, 20)

        var_dict = collections.OrderedDict()
        self.random_variables_block(64, 128, 128, 3, 1, var_dict, 20, 20)
        x = add_block(x, var_dict, 2, 1, 1, 0, 20, 20)
        
#         x = tf.nn.batch_normalization(x, b2_bn_mean, b2_bn_variance, b2_bn_offset, b2_bn_scale, BN_EPS, name='b2_bn')
        
#         x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='maxpool3')
        
#         x = conv2d(x, conv_new3_W, conv_new3_b, 2, 2, 2, 2, name='conv_new32')
#         x = tf.nn.relu(x, name='relu_new3')

        var_dict = collections.OrderedDict()
        self.random_variables_block(128, 128, 128, 3, 1, var_dict, 30, 30)
        x = add_block(x, var_dict, 2, 1, 1, 0, 30, 30)
        
        x = tf.layers.average_pooling2d(x, pool_size = 16, strides = 1, padding = "VALID", name='avgpool')
        
#         x = tf.reshape(x, [-1, 16384], name='flattening')
        x = tf.reshape(x, [-1, 128], name='flattening')
        
        x = tf.add(tf.matmul(x, dense1_weight), dense1_bias, name='dense1')
        x = tf.nn.relu(x, name='relu3')
        tf.add(tf.matmul(x, dense2_weight), dense2_bias, name='logits')
    
    
    ###########################################################################################################
    ### *****
    def define_loss_bodypath(self, opts):
        # Get graph:
        graph = tf.get_default_graph()
        # Inputs:
        y_pred = graph.get_tensor_by_name('logits:0')
        y_true = tf.placeholder(tf.float32, shape=(None, 1000), name='y_true')
        # Loss computation:
        softmax_cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        tf.divide(softmax_cross_entropy, tf.cast((opts.batch_size * 1000), dtype=tf.float32), name='loss')
#         tf.reduce_sum(tf.square(y_true-y_pred), name='loss')


    ###########################################################################################################
    ### Initialize variables of a batch normalization layer:
    def bn_variables(self, size, basename):
        # We initialize these variable in a fashion such that this layer will be the identity transformation (no effect).
        initial_zeros = tf.zeros((size))
        initial_ones = tf.ones((size))
        bn_mean = tf.Variable(initial_zeros, name=basename+'bn_mean')
        bn_variance = tf.Variable(initial_ones, name=basename+'bn_variance')
        bn_offset = tf.Variable(initial_zeros, name=basename+'bn_offset')
        bn_scale = tf.Variable(initial_ones, name=basename+'bn_scale')
        return bn_mean, bn_variance, bn_offset, bn_scale


    ########################################################################################
    #### Define optimizer:
    def define_optimizer(self, opts):
        # Get graph:
        graph = tf.get_default_graph()
        
        # Define optimizer:
        if opts.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(opts.initial_learning_rate)
        elif opts.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(opts.initial_learning_rate)
        elif opts.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(opts.initial_learning_rate, opts.momentum)
        elif opts.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(opts.initial_learning_rate, momentum=opts.momentum)
        else:
            tools.error('Optimizer not recognized.')
        
        # Operation to compute the gradients:
        loss = graph.get_tensor_by_name('loss:0')
        gradients = optimizer.compute_gradients(loss)
        
        # Operation to apply the gradietns:
        optimizer.apply_gradients(gradients, name='apply_grads_adam')
        
        return gradients


    ###########################################################################################################
    ### Create random variables for one block of the form conv + relu + conv + bn + relu
    def random_variables_block(self, dim_in, dim_mid, dim_out, kernelW, kernelH, var_dict, path_id, block_id):
        # Base name, including path id and block id:
        basename = 'p' + str(path_id) + '_b' + str(block_id) + '_'
        # Create variables and add them to dictionary:
        var_dict[basename + 'conv1_W'] = weight_variable([kernelW, kernelH, dim_in, dim_mid], name=basename+'conv1_W')
        var_dict[basename + 'conv1_b'] = bias_variable([dim_mid], name=basename+'conv1_b')
        var_dict[basename + 'conv2_W'] = weight_variable([kernelH, kernelW, dim_mid, dim_out], name=basename+'conv2_W')
        var_dict[basename + 'conv2_b'] = bias_variable([dim_out], name=basename+'conv2_b')

        var_dict[basename + 'bn_mean'], \
            var_dict[basename + 'bn_variance'], \
            var_dict[basename + 'bn_offset'], \
            var_dict[basename + 'bn_scale'] = self.bn_variables(dim_out, basename=basename)












