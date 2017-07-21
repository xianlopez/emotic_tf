### Main program. To be executed.
### Xian Lopez Alvarez
### 19/7/2017

import cnn_actions
import model_builder
import tools
import logging
import tensorflow as tf
import time
from data_loader import data_loader
# from tensorflow.python import debug as tf_debug

# Load options:
from options import opts

# Create directory to store results:
dircase = tools.prepare_dircase(opts)

# Write options to text file:
tools.write_options(opts, dircase)

# Configure the logger:
tools.configure_logging(dircase)

# Postprocess options, adjusting some values and looking for incompatibilities.
tools.postprocess_options(opts)

# Limit GPU's memory usage by tensorflow:
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = opts.memory_limit
    
# Load annotations:
annotations_train = tools.load_annotations('train')
annotations_val = tools.load_annotations('val')
annotations_test = tools.load_annotations('test')

# Create data loaders:
data_train = data_loader(annotations_train, opts)
data_val = data_loader(annotations_val, opts)
data_test = data_loader(annotations_test, opts)

# Create the computational graph, or import it:
if opts.restore_model:
    # Load meta-graph:
    checkpoint = tools.get_checkpoint(opts)
    metamodel_file = checkpoint + '.meta'
    saver = tf.train.import_meta_graph(metamodel_file)
else:
    # Create network and loss:
    gradients = model_builder.build_model(opts)
    saver = tf.train.Saver()

# Start clock:
start = time.time()

with tf.Session(config=tf_config) as sess:

    if opts.restore_model:
        checkpoint = tools.get_checkpoint(opts)
        saver.restore(sess, checkpoint)
    else:
        sess.run(tf.global_variables_initializer())
    
    # Train:
    if opts.train:
        cnn_actions.train(sess, saver, opts, dircase, data_train, data_val, gradients)
    
    # Do evaluation:
    if opts.evaluate_model:
        cnn_actions.evaluate_model(sess, opts, data_train, data_val, data_test)

logging.info('Process finished')






