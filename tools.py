### Global variables and tools for this project.
### Xian Lopez Alvarez
### 12/7/2017

import numpy as np
from PIL import Image
#from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import tensorflow as tf
import collections
from params import emotic_mean, emotic_std, category_names
import os
import time
import logging
import sys


###########################################################################################################
### Load annotations
def load_annotations(dataset = 'all'):
    
    if(dataset == 'train'):
        subset_name = '/home/xian/EMOTIC/train_annotations.txt'
    elif(dataset == 'val'):
        subset_name = '/home/xian/EMOTIC/val_annotations.txt'
    elif(dataset == 'test'):
        subset_name = '/home/xian/EMOTIC/test_annotations.txt'
    elif(dataset != 'all'):
        error('dataset not recognized.')
    
    with open('/home/xian/EMOTIC/annotations_txt.txt', "r") as f:
        annotations = list()
        for line in f:
            line_split = line.split(' ')
            path_rel = line_split[0]
            path_abs = '/home/xian/EMOTIC/EmpathyDB_images/' + path_rel
            person_index = line_split[1]
            x1 = line_split[2]
            y1 = line_split[3]
            x2 = line_split[4]
            y2 = line_split[5]
            valence = line_split[6]
            arousal = line_split[7]
            dominance = line_split[8]
            ncat = len(line_split) - 10
            categories = list()
            for idx in range(ncat):
                categories.append(line_split[9 + idx])
            cur_annotation = [path_abs, person_index, x1, y1, x2, y2, valence, arousal, dominance]
            cur_annotation.extend(categories)
            annotations.append(cur_annotation)
    
    if dataset != 'all':
        with open(subset_name, "r") as f:
            all_lines = f.readlines()
            nlines = len(all_lines)
            selected_indexes = np.zeros(nlines, dtype=np.int32)
            for idx in range(nlines):
                selected_indexes[idx] = np.int32(all_lines[idx][0:len(all_lines[idx])-1]) - 1
            
        # If we are not selecting the whole dataset, we keep only the corresponding annotations.
        aux = annotations
        annotations = list()
        for idx in range(nlines):
            annotations.append(aux[selected_indexes[idx]])            
            
    return annotations


###########################################################################################################
### Load full and body images, both in png format.
def load_images(annotation, preprocess):
    
    filename = annotation[0]
    person_idx = annotation[1]
    
    # Paths for full and body images:
    base_path = filename[34:len(filename)-4]
    base_path = '/home/xian/EMOTIC/images_preprocessed' + base_path
    
    path_full = base_path + '.png'
    path_body = base_path + '_body_' + str(person_idx) + '.png'
    
    # Load images:
    im_full = Image.open(path_full)
    im_body = Image.open(path_body)

    # Convert to numpy arrays:
    im_full_np = np.float32(im_full) / 255
    im_body_np = np.float32(im_body) / 255
    
    # Normalize:
    if preprocess == 0:
        # No normalization
        pass
    elif preprocess == 1:
        # Full normalization
        for idx_channel in range(3):
            im_full_np[:,:,idx_channel] = im_full_np[:,:,idx_channel] - emotic_mean[idx_channel]
            im_full_np[:,:,idx_channel] = im_full_np[:,:,idx_channel] / emotic_std[idx_channel]
            im_body_np[:,:,idx_channel] = im_body_np[:,:,idx_channel] - emotic_mean[idx_channel]
            im_body_np[:,:,idx_channel] = im_body_np[:,:,idx_channel] / emotic_std[idx_channel]
    elif preprocess == 2:
        # Normalization only with mean
        for idx_channel in range(3):
            im_full_np[:,:,idx_channel] = im_full_np[:,:,idx_channel] - emotic_mean[idx_channel]
            im_body_np[:,:,idx_channel] = im_body_np[:,:,idx_channel] - emotic_mean[idx_channel]
    else:
        error('preprocess option not understood.')
    
    return im_full_np, im_body_np


###########################################################################################################
### Get the categories that an annotation contains, in one-hot encoding.
def get_categories(annotation):
    categories_one_hot = np.zeros(26, dtype=np.int32)
    ncategories = len(annotation) - 9
    for cat_idx1 in range(ncategories):
        for cat_idx2 in range(26):
            if annotation[9 + cat_idx1] == category_names[cat_idx2]:
                categories_one_hot[cat_idx2] = 1
                break
    return categories_one_hot


###########################################################################################################
### Display error message, and stop execution.
def error(msg):
    logging.error('ERROR: ' + msg)
    exit()


###########################################################################################################
### Check a category is in a given annotation:
def category_in_annotation(annotation, idx_cat0):
    categories = annotation[9:len(annotation)]
    category_is_in = False
    for idx_cat1 in range(len(categories)):
        if categories[idx_cat1] == category_names[idx_cat0]:
            category_is_in = True
            break
    return category_is_in


###########################################################################################################
### Plot loss and metrics over the training process.
def plot_train(batches_train, batches_val, loss_train, loss_val, metrics_train, metrics_val, metric_names, dircase, opts):
    # batches_train: number of batch of each train measurement.
    # batches_val: number of batch of each validation measurement.
    # loss_train: loss in the training batches. len(loss_train) = len(batches_train)
    # loss_val: loss in the validation set, over the training. len(loss_val) = len(batches_val)
    # metrics_train: numpy array of shape (npoints_train, nmetrics)
    # metrics_val: numpy array of shape (npoints_val, nmetrics)
    
#    # Number of points on which we have train measurements:
#    npoints_train = len(batches_train)
#    # Number of points on which we have validation measurements:
#    npoints_val = len(batches_val)
    # Number of different metrics
    nmetrics = metrics_train.shape[1]
    
    colors = ['r', 'g', 'k', 'm', 'c', 'y', 'w']
    
    # Initialize figure:
    # Axis 1 will be for metrics, and axis 2 for losses.
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    # Plot train loss
    ax1.plot(batches_train, np.log(loss_train), 'b-', label='train loss')
    # Plot validation loss
    ax1.plot(batches_val, np.log(loss_val), 'b--', label='validation loss')
    
    for metric_idx in range(nmetrics):
        # Plot train metric:
        ax2.plot(batches_train, metrics_train[:,metric_idx], colors[metric_idx] + '-', label = 'train ' + metric_names[metric_idx])
        # Plot validation metric:
        ax2.plot(batches_val, metrics_val[:,metric_idx], colors[metric_idx] + '--', label = 'validation ' + metric_names[metric_idx])
    

    ax2.set_ylim(0, np.max((np.max(metrics_train), np.max(metrics_val))) * 1.1)
    plt.title('Model training history')
    ax1.set_ylabel('log Loss')
    ax2.set_ylabel('Metric')
    ax1.set_xlabel('Batch')
    
    # Add legend
    plt.legend(loc='upper left')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    fig.savefig(dircase + '/training_history.png')
    
    if opts.show_training_history:
        plt.show()


###########################################################################################################
### ****
def get_subdictionary(ini, length, full_dict):
    fin = ini + length
    sub_dict = collections.OrderedDict({})
    count = -1
    for k in full_dict:
        count = count + 1
        if count >= ini and count < fin:
            sub_dict[k] = full_dict[k]
    return sub_dict


###########################################################################################################
### Create directory to store results.
def prepare_dircase(opts):
    # Get today's date:
    date = time.strftime('%d/%m/%Y')
    date_split = date.split('/')
    date_day = date_split[0]
    date_month = date_split[1]
    date_year = date_split[2]
    # Check if the directory exists; if not, create it:
    dirdate = opts.dirbase + 'experiments/day_' + date_year + '_' + date_month + '_' + date_day
    if not os.path.exists(dirdate):
        os.makedirs(dirdate)
    # Look for the last experiment done today, and create a new one:
    case_idx = 1
    dircase = dirdate + '/case_' + str(case_idx)
    while os.path.exists(dircase):
        case_idx = case_idx + 1
        dircase = dirdate + '/case_' + str(case_idx)
    os.makedirs(dircase)
    print('Saving results to ' + dircase)
    return dircase


###########################################################################################################
### ****
def configure_logging(dircase):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=dircase+'/out.log',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logging.info('Logging configured.')


###########################################################################################################
### Write options to a file in the case directory.
def write_options(opts, dircase):
    with open(dircase+'/options.txt', 'w') as f:
        f.write('-----------------\n')
        f.write('General options:\n')
        f.write('-----------------\n')
        for key, val in vars(opts).items():
            if key != 'cnn_opts':
                f.write('%s: %s\n' % (key, str(val)))
        f.write('-----------------\n')
        f.write('CNN options\n')
        f.write('-----------------\n')
        for key, val in vars(opts.cnn_opts[opts.modelname]).items():
            f.write('%s: %s\n' % (key, str(val)))


###########################################################################################################
### Get checkpoint to restore a model.
def get_checkpoint(opts):
    if opts.checkpoint == 'last':
        checkpoint = tf.train.latest_checkpoint(opts.dir_saved_model)
    else:
        checkpoint = opts.dir_saved_model + '/model-' + str(opts.checkpoint)
    return checkpoint


###########################################################################################################
### Postprocess options, adjusting some values and looking for incompatibilities.
def postprocess_options(opts):
    # Supress random effects, if asked to:
    if opts.supress_random:
        np.random.seed(1)
        tf.set_random_seed(1)
        opts.keep_prob_train = 1
        opts.shuffle = False
    # It is not possible, by now, to debug and load a saved model.
    if opts.debug and opts.restore_model:
        error('Not possible, by now, to debug and load a saved model.')








