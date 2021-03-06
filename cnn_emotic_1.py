# CNN model for EMOTIC.

import tensorflow as tf
import collections
import numpy as np
from params import NDIM_DISC, NDIM_CONT, BN_EPS, STD_VAR_INI, NCAT_PLACES, NCAT_IMAGENET
import tools


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
    paddings = tf.constant([[0, 0], [padW, padW], [padH, padH], [0, 0]])
    return tf.add(tf.nn.conv2d(tf.pad(x, paddings), W, strides = [1, strideW, strideH, 1], padding = "VALID"), b, name=name)


###########################################################################################################
### Add one block of the form conv + relu + conv + bn + relu
def add_block(x_in, var_dict, strideW, strideH, padW, padH, path_id, block_id):
    # Base name, including path id and block id:
    basename = 'p' + str(path_id) + '_b' + str(block_id) + '_'
    # Block operations:
    conv1 = conv2d(x_in, var_dict.items()[0][1], var_dict.items()[1][1], strideW, strideH, padW, padH, name=basename+'conv1')
    relu1 = tf.nn.relu(conv1, name=basename+'relu1')
    conv2 = conv2d(relu1, var_dict.items()[2][1], var_dict.items()[3][1], strideH, strideW, padH, padW, name=basename+'conv2')
    bn = tf.nn.batch_normalization(conv2, var_dict.items()[4][1], var_dict.items()[5][1], \
        var_dict.items()[6][1], var_dict.items()[7][1], BN_EPS, name=basename+'bn')
    relu2 = tf.nn.relu(bn, name=basename+'relu2')
    return relu2


###########################################################################################################
### Add one block of the form conv + relu + conv + relu + bn
def add_block_v2(x_in, var_dict, strideW, strideH, padW, padH, path_id, block_id):
    # Base name, including path id and block id:
    basename = 'p' + str(path_id) + '_b' + str(block_id) + '_'
    # Block operations:
    conv1 = conv2d(x_in, var_dict.items()[0][1], var_dict.items()[1][1], strideW, strideH, padW, padH, name=basename+'conv1')
    relu1 = tf.nn.relu(conv1, name=basename+'relu1')
    conv2 = conv2d(relu1, var_dict.items()[2][1], var_dict.items()[3][1], strideH, strideW, padH, padW, name=basename+'conv2')
    relu2 = tf.nn.relu(conv2, name=basename+'relu2')
    bn = tf.nn.batch_normalization(relu2, var_dict.items()[4][1], var_dict.items()[5][1], \
        var_dict.items()[6][1], var_dict.items()[7][1], BN_EPS, name=basename+'bn')
    return bn


###########################################################################################################
### Load variables for one block of the form conv + relu + conv + bn + relu
def load_variables_block(dirbase, layer_ids, var_dict, path_id, block_id):
    # Base name, including path id and block id:
    basename = 'p' + str(path_id) + '_b' + str(block_id) + '_'
    # Load weights:
    [conv1_W, conv1_b] = read_weights_conv(dirbase+'l'+layer_ids[0]+'_conv.txt', varname=basename+'conv1')
    [conv2_W, conv2_b] = read_weights_conv(dirbase+'l'+layer_ids[1]+'_conv.txt', varname=basename+'conv2')
    [bn_mean, bn_variance, bn_offset, bn_scale] = \
        read_weights_bn(dirbase+'l'+layer_ids[2]+'_bn.txt', varname=basename+'bn')
    # Add variables to dictionary:
    var_dict[basename + 'conv1_W'] = conv1_W
    var_dict[basename + 'conv1_b'] = conv1_b
    var_dict[basename + 'conv2_W'] = conv2_W
    var_dict[basename + 'conv2_b'] = conv2_b
    var_dict[basename + 'bn_mean'] = bn_mean
    var_dict[basename + 'bn_variance'] = bn_variance
    var_dict[basename + 'bn_offset'] = bn_offset
    var_dict[basename + 'bn_scale'] = bn_scale


########################################################################################
#### ****
def read_weights_conv(filename, varname):
    # Get the dimensionality:
    with open(filename, "r") as f:
        dim1 = 0
        dim2 = 0
        dim3 = 0
        dim4 = 0
        dim2_done = False
        dim3_done = False
        for line in f:
            line_split = line.split(' ')
            if dim1 == 0:
                dim1 = len(line_split) - 1
            if line == 'enddim1\n' and not dim2_done:
                dim2 = dim2 + 1
            if line == 'enddim2\n' and not dim3_done:
                dim2_done = True
                dim3 = dim3 + 1
            if line == 'enddim3\n':
                dim3_done = True
                dim4 = dim4 + 1
            if line == 'enddim4\n':
                break
    
    dim_reord = [dim3, dim4, dim2, dim1]
    
    # Allocate variable for weights:
    weights = np.zeros(dim_reord)
    bias = np.zeros(dim_reord[3])
    
    # Load weights:
    with open(filename, "r") as f:
        # read weights
        for idx1 in range(dim_reord[0]):
            for idx2 in range(dim_reord[1]):
                for idx3 in range(dim_reord[2]):
                    more_read = True
                    while more_read:
                        line = f.readline()
                        line_split = line.split(' ')
                        more_read = len(line_split) == 1
                    for idx4 in range(dim_reord[3]):
                        weights[idx1, idx2, idx3, idx4] = float(line_split[idx4])
        # read bias
        more_read = True
        while more_read:
            line = f.readline()
            line_split = line.split(' ')
            more_read = len(line_split) == 1
        for idx1 in range(dim_reord[3]):
            bias[idx1] = float(line_split[idx1])
    
    tensor_weights = tf.Variable(np.float32(weights), name=varname+'_W')
    tensor_bias = tf.Variable(np.float32(bias), name=varname+'_b')
    
    return tensor_weights, tensor_bias


########################################################################################
#### ****
def read_weights_bn(filename, varname):
    with open(filename, "r") as f:
        # bias
        line_bias = f.readline()
        line_split = line_bias.split(' ')
        nunits = len(line_split) - 1 # get the dimensionality
        bias = np.zeros(nunits)
        for idx in range(nunits):
            bias[idx] = float(line_split[idx])
        # weight
        line_weight = f.readline()
        line_split = line_weight.split(' ')
        weight = np.zeros(nunits)
        for idx in range(nunits):
            weight[idx] = float(line_split[idx])
        # running variance
        line_running_var = f.readline()
        line_split = line_running_var.split(' ')
        running_var = np.zeros(nunits)
        for idx in range(nunits):
            running_var[idx] = float(line_split[idx])
        # running mean
        line_running_mean = f.readline()
        line_split = line_running_mean.split(' ')
        running_mean = np.zeros(nunits)
        for idx in range(nunits):
            running_mean[idx] = float(line_split[idx])
    
    tensor_running_mean = tf.Variable(np.float32(running_mean), name=varname+'_mean')
    tensor_running_var = tf.Variable(np.float32(running_var), name=varname+'_variance')
    tensor_bias = tf.Variable(np.float32(bias), name=varname+'_offset')
    tensor_weight = tf.Variable(np.float32(weight), name=varname+'_scale')
    
    return tensor_running_mean, tensor_running_var, tensor_bias, tensor_weight


########################################################################################
#### ****
def read_weights_dense(filename, varname):
    # Get the dimensionality:
    with open(filename, "r") as f:
        dimin = -1
        dimout = 0
        for line in f:
            dimin = dimin + 1
            if dimout == 0:
                line_split = line.split(' ')
                dimout = len(line_split) - 1
    
    # Load weights:
    with open(filename, "r") as f:
        f = open(filename, "r")
        # bias (first line)
        line_bias = f.readline()
        line_split = line_bias.split(' ')
        bias = np.zeros(dimout)
        for idx in range(dimout):
            bias[idx] = float(line_split[idx])
        # weight (rest of lines)
        weight = np.zeros((dimin, dimout))
        for idx1 in range(dimin):
            line_weight = f.readline()
            line_split = line_weight.split(' ')
            for idx2 in range(dimout):
                weight[idx1, idx2] = float(line_split[idx2])
    
    tensor_weight = tf.Variable(np.float32(weight), name=varname+'_weight')
    tensor_bias = tf.Variable(np.float32(bias), name=varname+'_bias')
    
    return tensor_weight, tensor_bias


########################################################################################
#### Main class
class cnn_builder_class:
    # Specific options for this network:
    load_torch = False
    dirmodel = []
    correct_block2 = False
    correct_avgpool = False
    batch_size = 0
    xavier_init = False


    ########################################################################################
    #### Load random weights for the full path:
    def __init__(self, cnn_opts, opts):
        self.load_torch = cnn_opts.load_torch
        self.dirmodel = cnn_opts.dirmodel
        self.correct_block2 = cnn_opts.correct_block2
        self.correct_avgpool = cnn_opts.correct_avgpool
        self.batch_size = opts.batch_size
        self.xavier_init = opts.xavier_init


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


    ###########################################################################################################
    ### Create random variables for one block of the form conv + relu + conv + bn + relu
    def random_variables_block(self, dim_in, dim_mid, dim_out, kernelW, kernelH, var_dict, path_id, block_id):
        # Base name, including path id and block id:
        basename = 'p' + str(path_id) + '_b' + str(block_id) + '_'
        # Create variables and add them to dictionary:
        if self.xavier_init:
            var_dict[basename + 'conv1_W'] = weight_variable_xavier([kernelW, kernelH, dim_in, dim_mid], dim_in*kernelW*kernelH, dim_mid*kernelW*kernelH, name=basename+'conv1_W')
            var_dict[basename + 'conv1_b'] = bias_variable([dim_mid], name=basename+'conv1_b')
            var_dict[basename + 'conv2_W'] = weight_variable_xavier([kernelH, kernelW, dim_mid, dim_out], dim_mid*kernelW*kernelH, dim_out*kernelW*kernelH, name=basename+'conv2_W')
            var_dict[basename + 'conv2_b'] = bias_variable([dim_out], name=basename+'conv2_b')
#             var_dict[basename + 'bn_mean'] = bias_variable([dim_out], name=basename+'bn_mean')
#             var_dict[basename + 'bn_variance'] = weight_variable_xavier([dim_out], dim_out, dim_out, name=basename+'bn_variance')
#             var_dict[basename + 'bn_offset'] = bias_variable([dim_out], name=basename+'bn_offset')
#             var_dict[basename + 'bn_scale'] = weight_variable_xavier([dim_out], dim_out, dim_out, name=basename+'bn_scale')
        else:
            var_dict[basename + 'conv1_W'] = weight_variable([kernelW, kernelH, dim_in, dim_mid], name=basename+'conv1_W')
            var_dict[basename + 'conv1_b'] = bias_variable([dim_mid], name=basename+'conv1_b')
            var_dict[basename + 'conv2_W'] = weight_variable([kernelH, kernelW, dim_mid, dim_out], name=basename+'conv2_W')
            var_dict[basename + 'conv2_b'] = bias_variable([dim_out], name=basename+'conv2_b')
#             var_dict[basename + 'bn_mean'] = weight_variable([dim_out], name=basename+'bn_mean')
#             var_dict[basename + 'bn_variance'] = weight_variable([dim_out], name=basename+'bn_variance')
#             var_dict[basename + 'bn_offset'] = weight_variable([dim_out], name=basename+'bn_offset')
#             var_dict[basename + 'bn_scale'] = weight_variable([dim_out], name=basename+'bn_scale')
        var_dict[basename + 'bn_mean'], \
            var_dict[basename + 'bn_variance'], \
            var_dict[basename + 'bn_offset'], \
            var_dict[basename + 'bn_scale'] = self.bn_variables(dim_out, basename=basename)


    ########################################################################################
    #### Load random weights for the full path:
    def random_weights_full(self):
        # Initialize dictionary:
        var_dict = collections.OrderedDict()
        # block 1
        self.random_variables_block(3, 32, 64, 1, 11, var_dict, 1, 1)
        # block 2
        self.random_variables_block(64, 128, 256, 1, 5, var_dict, 1, 2)
        # block 3
        self.random_variables_block(256, 384, 512, 1, 3, var_dict, 1, 3)
        # block 4
        self.random_variables_block(512, 384, 384, 1, 3, var_dict, 1, 4)
        # block 5
        self.random_variables_block(384, 640, 640, 1, 3, var_dict, 1, 5)
        # block 6
        self.random_variables_block(640, 640, 640, 1, 3, var_dict, 1, 6)
        # block 7
        self.random_variables_block(640, 640, 640, 1, 3, var_dict, 1, 7)
        # block 8
        self.random_variables_block(640, 640, 640, 1, 3, var_dict, 1, 8)
        return var_dict


    ########################################################################################
    #### Load weights from Torch model for the full path:
    def load_weights_full(self):
        # Initialize dictionary:
        var_dict = collections.OrderedDict()
        # Directory where the weights are:
        dirbase = self.dirmodel + 'tramo1/'
        # block 1
        load_variables_block(dirbase, ['1', '3', '4'], var_dict, 1, 1)
        # block 2
        load_variables_block(dirbase, ['6', '8', '9'], var_dict, 1, 2)
        # block 3
        load_variables_block(dirbase, ['11', '13', '14'], var_dict, 1, 3)
        # block 4
        load_variables_block(dirbase, ['16', '18', '19'], var_dict, 1, 4)
        # block 5
        load_variables_block(dirbase, ['21', '23', '24'], var_dict, 1, 5)
        # block 6
        load_variables_block(dirbase, ['26', '28', '29'], var_dict, 1, 6)
        # block 7
        load_variables_block(dirbase, ['31', '33', '34'], var_dict, 1, 7)
        # block 8
        load_variables_block(dirbase, ['36', '38', '39'], var_dict, 1, 8)
        return var_dict


    ########################################################################################
    #### Load random weights for the body path:
    def random_weights_body(self):
        # Initialize dictionary:
        var_dict = collections.OrderedDict()
        # block 1
        self.random_variables_block(3, 32, 64, 1, 3, var_dict, 2, 1)
        # block 2
        self.random_variables_block(64, 128, 128, 1, 3, var_dict, 2, 2)
        # block 3
        self.random_variables_block(128, 128, 128, 1, 3, var_dict, 2, 3)
        return var_dict


    ########################################################################################
    #### Load weights from Torch model for the body path:
    def load_weights_body(self):
        # Initialize dictionary:
        var_dict = collections.OrderedDict()
        # Directory where the weights are:
        dirbase = self.dirmodel + 'tramo2/'
        # block 1
        load_variables_block(dirbase, ['1', '3', '4'], var_dict, 2, 1)
        # block 2
        load_variables_block(dirbase, ['6', '8', '10'], var_dict, 2, 2)
        # block 3
        load_variables_block(dirbase, ['11', '13', '14'], var_dict, 2, 3)
        return var_dict


    ########################################################################################
    #### Define the full image path of the network:
    def full_path(self):
        # Full image path:
        x_f = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x_f')

        # Parameters
        if self.load_torch:
            if self.dirmodel == None:
                tools.error('Missing dirmodel.')
            var_dict = self.load_weights_full()
        else:
            var_dict = self.random_weights_full()

        # Model
        block_configurations = [[4, 1, 2, 0],
            [2, 1, 2, 0],
            [2, 1, 1, 0],
            [1, 1, 1, 0],
            [2, 1, 1, 0],
            [1, 1, 1, 0],
            [2, 1, 1, 0],
            [1, 1, 1, 0]]
        # Loop adding all blocks:
        ini = 0
        x = x_f
        for block_idx in range(8):
            var_dict_block = tools.get_subdictionary(ini, 8, var_dict)
            strideW = block_configurations[block_idx][0]
            strideH = block_configurations[block_idx][1]
            padW = block_configurations[block_idx][2]
            padH = block_configurations[block_idx][3]
            x = add_block(x, var_dict_block, strideW, strideH, padW, padH, 1, block_idx+1)
            ini = ini + 8
        # average pooling
        p1_avgpool = tf.layers.average_pooling2d(x, pool_size = 4, strides = 1, padding = "VALID", name='p1_avgpool')
        # flattening
        full_path_end = tf.reshape(p1_avgpool, [-1, 640], name='full_path_end')
        
        return full_path_end


    ########################################################################################
    #### Define the body image path of the network:
    def body_path(self):
        # Full image path:
        x_b = tf.placeholder(tf.float32, shape=[None, 128, 128, 3], name='x_b')

        # Check we are not applying any correction if loading the Torch model:
        if self.load_torch and (self.correct_block2 or self.correct_avgpool):
            tools.error('Not posible to load Torch model and apply any correction.')

        # Parameters
        if self.load_torch:
            if self.dirmodel == None:
                tools.error('Missing dirmodel.')
            var_dict = self.load_weights_body()
        else:
            var_dict = self.random_weights_body()

        # Model
        block_configurations = [[2, 1, 1, 0],
            [2, 1, 1, 0],
            [2, 1, 1, 0]]
        # Loop adding all blocks:
        ini = 0
        x = x_b
        for block_idx in range(3):
            var_dict_block = tools.get_subdictionary(ini, 8, var_dict)
            strideW = block_configurations[block_idx][0]
            strideH = block_configurations[block_idx][1]
            padW = block_configurations[block_idx][2]
            padH = block_configurations[block_idx][3]
            if (not self.correct_block2) and block_idx == 1:
                x = add_block_v2(x, var_dict_block, strideW, strideH, padW, padH, 2, block_idx+1)
            else:
                x = add_block(x, var_dict_block, strideW, strideH, padW, padH, 2, block_idx+1)
            ini = ini + 8
        # average pooling
        if self.correct_avgpool:
            p2_avgpool = tf.layers.average_pooling2d(x, pool_size = 16, strides = 1, padding = "VALID", name='p2_avgpool')
        else:
            p2_avgpool = tf.layers.average_pooling2d(x, pool_size = 3, strides = 16, padding = "VALID", name='p2_avgpool')
        # flattening
        body_path_end = tf.reshape(p2_avgpool, [-1, 128], name='body_path_end')
        
        return body_path_end


    ########################################################################################
    #### Define the joint path of the network:
    def joint_path(self, p1_flat, p2_flat):

        # Parameters
        if self.load_torch:
            if self.dirmodel == None:
                tools.error('Missing dirmodel.')
            dirbase = self.dirmodel + 'tramo3/'
            [pj_dense_weight, pj_dense_bias] = read_weights_dense(dirbase+'l1_dense.txt', varname='pj_dense')
            [pj_bn_mean, pj_bn_variance, pj_bn_offset, pj_bn_scale] = \
                read_weights_bn(dirbase+'l2_bn.txt', varname='pj_bn')
            
        else:
            if self.xavier_init:
                pj_dense_weight = weight_variable_xavier([768, 256], 768, 256, name='pj_dense_weight')
                pj_dense_bias = bias_variable([256], name='pj_dense_bias')
#                 pj_bn_mean = bias_variable([256], name='pj_bn_mean')
#                 pj_bn_variance = weight_variable_xavier([256], 256, 256, name='pj_bn_variance')
#                 pj_bn_offset = bias_variable([256], name='pj_bn_offset')
#                 pj_bn_scale = weight_variable_xavier([256], 256, 256, name='pj_bn_scale')
            else:
                pj_dense_weight = weight_variable([768, 256], name='pj_dense_weight')
                pj_dense_bias = bias_variable([256], name='pj_dense_bias')
#                 pj_bn_mean = weight_variable([256], name='pj_bn_mean')
#                 pj_bn_variance = weight_variable([256], name='pj_bn_variance')
#                 pj_bn_offset = weight_variable([256], name='pj_bn_offset')
#                 pj_bn_scale = weight_variable([256], name='pj_bn_scale')
            pj_bn_mean, \
                pj_bn_variance, \
                pj_bn_offset, \
                pj_bn_scale = self.bn_variables(256, basename='pj_bn_scale')
        
        # Probability of keeping units in the dropout layer:
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Model
        pj_join = tf.concat([p1_flat, p2_flat], 1, name='pj_join')
        pj_dense = tf.add(tf.matmul(pj_join, pj_dense_weight), pj_dense_bias, name='pj_dense')
        pj_bn = tf.nn.batch_normalization(pj_dense, pj_bn_mean, pj_bn_variance, pj_bn_offset, pj_bn_scale, BN_EPS, name='pj_bn')
        pj_relu1 = tf.nn.relu(pj_bn, name='pj_relu1')
        pj_dropout = tf.nn.dropout(pj_relu1, keep_prob, name='pj_dropout')
        
        return pj_dropout


    ########################################################################################
    #### Define the predictions paths of the network:
    def predictions_path(self, pj_dropout):

        # Parameters
        if self.load_torch:
            if self.dirmodel == None:
                tools.error('Missing dirmodel.')
            dirbase = self.dirmodel + 'tramo3/'
            [yc_weight, yc_bias] = read_weights_dense(dirbase+'l6_dense.txt', varname='yc')
            [yd_weight, yd_bias] = read_weights_dense(dirbase+'l7_dense.txt', varname='yd')
            
        else:
            if self.xavier_init:
                yc_weight = weight_variable_xavier([256, NDIM_CONT], 256, NDIM_CONT, name='yc_weight')
                yc_bias = bias_variable([NDIM_CONT], name='yc_bias')
                yd_weight = weight_variable_xavier([256, NDIM_DISC], 256, NDIM_DISC, name='yd_weight')
                yd_bias = bias_variable([NDIM_DISC], name='yd_bias')
            else:
                yc_weight = weight_variable([256, NDIM_CONT], name='yc_weight')
                yc_bias = bias_variable([NDIM_CONT], name='yc_bias')
                yd_weight = weight_variable([256, NDIM_DISC], name='yd_weight')
                yd_bias = bias_variable([NDIM_DISC], name='yd_bias')

        # Model
        yc = tf.add(tf.matmul(pj_dropout, yc_weight), yc_bias, name='yc')
        yd = tf.add(tf.matmul(pj_dropout, yd_weight), yd_bias, name='yd')
#         yc = tf.sigmoid(tf.add(tf.matmul(pj_dropout, yc_weight), yc_bias), name='yc')
#         yd = tf.sigmoid(tf.add(tf.matmul(pj_dropout, yd_weight), yd_bias), name='yd')

        return yc, yd


    ########################################################################################
    #### Define the predictions for the full image network:
    def predictions_full(self, full_path_end):
        
        if self.xavier_init:
            pred_dense1_weight = weight_variable_xavier([640, 256], 640, 256, name='pred_dense1_weight')
            pred_dense1_bias = bias_variable([256], name='pred_dense1_bias')
            pred_dense2_weight = weight_variable_xavier([256, NCAT_PLACES], 256, NCAT_PLACES, name='pred_dense2_weight')
            pred_dense2_bias = bias_variable([NCAT_PLACES], name='pred_dense2_bias')
        else:
            pred_dense1_weight = weight_variable([640, 256], name='pred_dense1_weight')
            pred_dense1_bias = bias_variable([256], name='pred_dense1_bias')
            pred_dense2_weight = weight_variable([256, NCAT_PLACES], name='pred_dense2_weight')
            pred_dense2_bias = bias_variable([NCAT_PLACES], name='pred_dense2_bias')

        # Model
        pred_dense1 = tf.add(tf.matmul(full_path_end, pred_dense1_weight), pred_dense1_bias, name='pred_dense1')
        y = tf.add(tf.matmul(pred_dense1, pred_dense2_weight), pred_dense2_bias, name='y')
        
        return y


    ########################################################################################
    #### Define the predictions for the body image network:
    def predictions_body(self, body_path_end):
        
        if self.xavier_init:
            pred_dense1_weight = weight_variable_xavier([128, 256], 128, 256, name='pred_dense1_weight')
            pred_dense1_bias = bias_variable([256], name='pred_dense1_bias')
            pred_dense2_weight = weight_variable_xavier([256, NCAT_IMAGENET], 256, NCAT_IMAGENET, name='pred_dense2_weight')
            pred_dense2_bias = bias_variable([NCAT_IMAGENET], name='pred_dense2_bias')
        else:
            pred_dense1_weight = weight_variable([640, 256], name='pred_dense1_weight')
            pred_dense1_bias = bias_variable([256], name='pred_dense1_bias')
            pred_dense2_weight = weight_variable([256, NCAT_IMAGENET], name='pred_dense2_weight')
            pred_dense2_bias = bias_variable([NCAT_IMAGENET], name='pred_dense2_bias')

        # Model
        pred_dense1 = tf.add(tf.matmul(body_path_end, pred_dense1_weight), pred_dense1_bias, name='pred_dense1')
        y = tf.add(tf.matmul(pred_dense1, pred_dense2_weight), pred_dense2_bias, name='y')
        
        return y


    ########################################################################################
    #### Define CNN EMOTIC:
    def define_network(self):
        # Full image path:
        full_path_end = self.full_path()
        # Body image path:
        body_path_end = self.body_path()
        # Joint path:
        pj_dropout = self.joint_path(full_path_end, body_path_end)
        # Predictions paths:
        yc, yd = self.predictions_path(pj_dropout)
        # Create a dictionary with all the input and output variables:
        graph = tf.get_default_graph()
        cnn_dict = {
            'yc'       : yc,
            'yd'       : yd,
            'x_f'      : graph.get_tensor_by_name('x_f:0'),
            'x_b'      : graph.get_tensor_by_name('x_b:0'),
            'keep_prob': graph.get_tensor_by_name('keep_prob:0')
        }
        return cnn_dict


    ########################################################################################
    #### Define network only with the full image path:
    def define_fullpath(self):
        # Full image path:
        full_path_end = self.full_path()
        # Predictions paths:
        y = self.predictions_full(full_path_end)
        # Create a dictionary with all the input and output variables:
        graph = tf.get_default_graph()
        cnn_dict = {
            'y'        : y,
            'x_f'        : graph.get_tensor_by_name('x_f:0'),
            'keep_prob': graph.get_tensor_by_name('keep_prob:0')
        }
        return cnn_dict


    ########################################################################################    ########################################################################################
    #### Define network only with the body image path:
    def define_bodypath(self):
        # Body image path:
        body_path_end = self.body_path()
        # Predictions paths:
        y = self.predictions_body(body_path_end)
        # Create a dictionary with all the input and output variables:
        graph = tf.get_default_graph()
        cnn_dict = {
            'y'        : y,
            'x_b'        : graph.get_tensor_by_name('x_b:0'),
            'keep_prob': graph.get_tensor_by_name('keep_prob:0')
        }
        return cnn_dict


    ########################################################################################
    #### Define original loss, as described in the paper and the Torch code:
    def define_loss_orig(self, opts):
        # Get graph:
        graph = tf.get_default_graph()
        # Inputs:
        y_pred_cont = graph.get_tensor_by_name('yc:0')
        y_pred_disc = graph.get_tensor_by_name('yd:0')
        y_true_cont = tf.placeholder(tf.float32, shape=(None, NDIM_CONT), name='y_true_cont')
        y_true_disc = tf.placeholder(tf.float32, shape=(None, NDIM_DISC), name='y_true_disc')

        # Continuous loss:
        dif_cont = tf.abs(y_pred_cont - y_true_cont)
        dif_cont_sq = tf.square(dif_cont)
        over_margin = tf.cast(tf.greater(dif_cont, opts.loss_cont_margin), dtype=tf.float32)
        saturated = tf.cast(tf.greater(dif_cont, opts.loss_cont_saturation), dtype=tf.float32)
        aux1_cont = dif_cont_sq * over_margin * (1 - saturated)
        aux2_cont = tf.zeros((self.batch_size, 3)) + tf.ones((self.batch_size, 3)) * saturated * opts.loss_cont_saturation
        aux3_cont = aux1_cont + aux2_cont
        L_cont = tf.divide(tf.reduce_sum(aux3_cont), tf.cast((self.batch_size * NDIM_CONT), dtype=tf.float32), name='L_cont')

        # Discrete loss:
        # Class weights:
#         if opts.reweight == 0:
#             ldisc_weights = tf.ones((NDIM_DISC), dtype=tf.float32, name='ldisc_weights')
#         elif opts.reweight == 1:
#             sum_classes = tf.reduce_sum(y_true_disc, axis=0, name='sum_classes')
#             aux0_disc = tf.cast(tf.greater(sum_classes, 0.5), dtype=tf.float32)
#             aux1_disc = tf.log(tf.add(sum_classes, opts.ldisc_c), name='aux1_disc')
#             ldisc_weights_prev = tf.divide(tf.ones((NDIM_DISC)), aux1_disc)
#             ldisc_weights = tf.multiply(ldisc_weights_prev, aux0_disc, name='ldisc_weights')
#         elif opts.reweight == 2:
#             tools.error('reweight = 2 not ready yet.')
#         else:
#             tools.error('reweight option not understood.')
        ldisc_weights = tf.placeholder(tf.float32, shape=(NDIM_DISC), name='ldisc_weights')
        # Rest of the computations:
        dif_disc_sq = tf.square(y_pred_disc - y_true_disc, name='dif_disc_sq')
        weights_expanded1 = tf.expand_dims(ldisc_weights, axis=0, name='weights_expanded1')
        aux2_disc = tf.cast(tf.ones(()) * self.batch_size, dtype=tf.int32)
        aux3_disc = tf.stack([aux2_disc, tf.cast(tf.ones(()), dtype=tf.int32)], axis=0, name='aux3_disc')
        weights_expanded2 = tf.tile(weights_expanded1, multiples=aux3_disc, name='weights_expanded2')
        aux4_disc = dif_disc_sq * weights_expanded2
        L_disc = tf.divide(tf.reduce_sum(aux4_disc), tf.cast((self.batch_size * NDIM_DISC), dtype=tf.float32), name='L_disc')

        # Combination:
        weighted_cont = opts.w_cont * L_cont
        weighted_disc = opts.w_disc * L_disc
        if opts.loss_type == 'orig':
            L_comb = tf.add(weighted_cont, weighted_disc, name='L_comb')
        elif opts.loss_type == 'onlycont':
            L_comb = tf.add(weighted_cont, 0, name='L_comb')
        elif opts.loss_type == 'onlydisc':
            L_comb = tf.add(weighted_disc, 0, name='L_comb')
        else:
            tools.error('Incorrect value for option loss_type.')

        # Return a dictionary with the loss output and all its parameters and inputs:
        loss_dict = {
            'L_comb'              : L_comb,
            'y_true_cont'         : y_true_cont,
            'y_true_disc'         : y_true_disc
        }
        return loss_dict


    ########################################################################################
    #### Define simpler loss:
    def define_loss_simple1(self):
        # Get graph:
        graph = tf.get_default_graph()
        # Inputs:
        y_pred_cont = graph.get_tensor_by_name('yc:0')
        y_pred_disc = graph.get_tensor_by_name('yd:0')
        y_true_cont = tf.placeholder(tf.float32, shape=(None, NDIM_CONT), name='y_true_cont')
        y_true_disc = tf.placeholder(tf.float32, shape=(None, NDIM_DISC), name='y_true_disc')

        # Parameters:
        w_cont = tf.placeholder(tf.float32, shape=(), name='w_cont') # defaults to 1
        w_disc = tf.placeholder(tf.float32, shape=(), name='w_disc') # defaults to 1/6
        loss_cont_margin = tf.placeholder(tf.float32, shape=(), name='loss_cont_margin') # defaults to 0
        loss_cont_saturation = tf.placeholder(tf.float32, shape=(), name='loss_cont_saturation') # defaults to 1500

        # Continuous loss:
        mse = tf.losses.mean_squared_error(labels=y_true_cont, predictions=y_pred_cont)
        L_cont = tf.divide(mse, tf.cast((self.batch_size * NDIM_DISC), dtype=tf.float32), name='L_cont')

        # Discrete loss:
        sigmoid_cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_disc, logits=y_pred_disc))
        L_disc = tf.divide(sigmoid_cross_entropy, tf.cast((self.batch_size * NDIM_DISC), dtype=tf.float32), name='L_disc')

        # Combination:
        weighted_cont = w_cont * L_cont
        weighted_disc = w_disc * L_disc
        L_comb = tf.add(weighted_cont, weighted_disc, name='L_comb')

        # Return a dictionary with the loss output and all its parameters and inputs:
        loss_dict = {
            'L_comb'              : L_comb,
            'y_true_cont'         : y_true_cont,
            'y_true_disc'         : y_true_disc,
            'w_cont'              : w_cont,
            'w_disc'              : w_disc,
            'loss_cont_margin'    : loss_cont_margin,
            'loss_cont_saturation': loss_cont_saturation
        }
        return loss_dict


    ########################################################################################
    #### Define loss when the network hast only the full image path:
    def define_loss_fullpath(self):
        # Get graph:
        graph = tf.get_default_graph()
        # Inputs:
        y_pred = graph.get_tensor_by_name('y:0')
        y_true = tf.placeholder(tf.float32, shape=(None, NCAT_PLACES), name='y_true')
        # Loss computation:
        sigmoid_cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        L_disc = tf.divide(sigmoid_cross_entropy, tf.cast((self.batch_size * NCAT_PLACES), dtype=tf.float32), name='L_disc')
        # Return a dictionary with the loss output and all its parameters and inputs:
        loss_dict = {
            'L_disc'              : L_disc,
            'y_true'         : y_true,
        }
        return loss_dict


    ########################################################################################
    #### Define loss when the network hast only the body image path:
    def define_loss_bodypath(self):
        # Get graph:
        graph = tf.get_default_graph()
        # Inputs:
        y_pred = graph.get_tensor_by_name('y:0')
        y_true = tf.placeholder(tf.float32, shape=(None, NCAT_IMAGENET), name='y_true')
        # Loss computation:
        sigmoid_cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        L_disc = tf.divide(sigmoid_cross_entropy, tf.cast((self.batch_size * NCAT_IMAGENET), dtype=tf.float32), name='L_disc')
        # Return a dictionary with the loss output and all its parameters and inputs:
        loss_dict = {
            'L_disc'              : L_disc,
            'y_true'         : y_true,
        }
        return loss_dict


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
        L_comb = graph.get_tensor_by_name('L_comb:0')
        gradients = optimizer.compute_gradients(L_comb)
        
        # Operation to apply the gradietns:
        optimizer.apply_gradients(gradients, name='apply_grads_adam')
        
        return gradients







