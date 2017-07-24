### Global variables and tools for this project.
### Xian Lopez Alvarez
### 12/7/2017

import numpy as np
from PIL import Image
#from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import tensorflow as tf
import collections
from params import emotic_mean, emotic_std, places_mean, places_std, imagenet_mean, imagenet_std, category_names
import os
import time
import logging
import sys


###########################################################################################################
### Load annotations
def load_annotations(dataset, opts):
    
    if(dataset == 'train'):
        subset_name = opts.dirbase + 'train_annotations.txt'
    elif(dataset == 'val'):
        subset_name = opts.dirbase + 'val_annotations.txt'
    elif(dataset == 'test'):
        subset_name = opts.dirbase + 'test_annotations.txt'
    elif(dataset != 'all'):
        error('dataset not recognized.')
    
    # Load full file with all the annotations:
    with open(opts.dirbase + 'annotations_txt.txt', "r") as f:
        annotations = list()
        for line in f:
            line_split = line.split(' ')
            path_rel = '/' + line_split[0]
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
            cur_annotation = [path_rel, person_index, x1, y1, x2, y2, valence, arousal, dominance]
            cur_annotation.extend(categories)
            annotations.append(cur_annotation)
    
    # Keep only those corresonding to the set we are interested in:
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
    
    # If mini_dataset options is on, keep only a portion of the annotations:
    if opts.mini_dataset:
        aux = annotations
        nprev = len(annotations)
        nmini = int(np.round(np.float32(nprev) * opts.mini_percent / 100.0))
        selected_indexes = np.random.choice(range(nprev), nmini, replace=False)
        annotations = list()
        for idx in range(nmini):
            annotations.append(aux[selected_indexes[idx]])
        logging.info('Number of annotations reduced from %i to %i.' % (nprev, nmini))
            
    return annotations


###########################################################################################################
### Load full and body images, both in png format.
def load_images(annotation, opts):
    
    filename = annotation[0]
    
    if opts.load_from_png:
        person_idx = annotation[1]
        # Paths for full and body images:
#         base_path = filename[34:len(filename)-4]
        base_path = filename[0:len(filename)-4]
        base_path = opts.emotic_dir_png + base_path
        path_full = base_path + '.png'
        path_body = base_path + '_body_' + str(person_idx) + '.png'
        # Load images:
        im_full = Image.open(path_full)
        im_body = Image.open(path_body)
        # Convert to numpy arrays:
        im_full_np = np.float32(im_full) / 255
        im_body_np = np.float32(im_body) / 255
    
    else:
        # Image path:
#         filename = filename[34:len(filename)]
        filename = filename
        fullpath = opts.emotic_dir_jpg + filename
        # Load image:
        im_orig = Image.open(fullpath)
        # Body bounding box:
        body_bbox = [int(annotation[2]), int(annotation[3]), int(annotation[4]), int(annotation[5])]
        # Convert to numpy arrays:
        im_full_np, im_body_np = preprocess_emotic(im_orig, body_bbox, opts)
    
    # Normalize:
    if opts.normalize == 0:
        # No normalization
        pass
    elif opts.normalize == 1:
        # Full normalization
        for idx_channel in range(3):
            im_full_np[:,:,idx_channel] = im_full_np[:,:,idx_channel] - emotic_mean[idx_channel]
            im_full_np[:,:,idx_channel] = im_full_np[:,:,idx_channel] / emotic_std[idx_channel]
            im_body_np[:,:,idx_channel] = im_body_np[:,:,idx_channel] - emotic_mean[idx_channel]
            im_body_np[:,:,idx_channel] = im_body_np[:,:,idx_channel] / emotic_std[idx_channel]
    elif opts.normalize == 2:
        # Normalization only with mean
        for idx_channel in range(3):
            im_full_np[:,:,idx_channel] = im_full_np[:,:,idx_channel] - emotic_mean[idx_channel]
            im_body_np[:,:,idx_channel] = im_body_np[:,:,idx_channel] - emotic_mean[idx_channel]
    else:
        error('normalize option not understood.')
    
    return im_full_np, im_body_np


###########################################################################################################
### Load image from ImageNet.
def load_images_onepath(filepath, opts):
    # Load image:
    image = Image.open(filepath)
    # Convert to numpy arrays:
    im_prep = preprocess_onepath(image, opts)
    # Select mean and standard deviation according to the dataset:
    if opts.net_arch == 'fullpath':
        data_mean = places_mean
        data_std = places_std
    elif opts.net_arch == 'bodypath':
        data_mean = imagenet_mean
        data_std = imagenet_std
    else:
        error('net_arch option nor recognized.')
    # Normalize:
    if opts.normalize == 0:
        # No normalization
        pass
    elif opts.normalize == 1:
        # Full normalization
        for idx_channel in range(3):
            im_prep[:,:,idx_channel] = im_prep[:,:,idx_channel] - data_mean[idx_channel]
            im_prep[:,:,idx_channel] = im_prep[:,:,idx_channel] / data_std[idx_channel]
    elif opts.normalize == 2:
        # Normalization only with mean
        for idx_channel in range(3):
            im_prep[:,:,idx_channel] = im_prep[:,:,idx_channel] - data_mean[idx_channel]
    else:
        error('normalize option not understood.')
    return im_prep


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
def plot_train(batches_train, batches_val, loss_train, loss_val, metrics_train, metrics_val, metric_names, dircase, show_training_history, global_step):
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
    
    if global_step == 'final':
        figname = dircase + '/training_history.png'
    else:
        figname = dircase + '/training_' + str(global_step) + '.png'
    
    fig.savefig(figname)
    
    if show_training_history:
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
        opts.randomize_preprocess = False
    elif opts.random_seed > 0:
        # Set seed
        np.random.seed(opts.random_seed)
        tf.set_random_seed(opts.random_seed)
    # It is not possible, by now, to debug and load a saved model.
    if opts.debug and opts.restore_model:
        error('Not possible, by now, to debug and load a saved model.')
    # If we choose to load image with png format, we cannot choose other dataset apart from EMOTIC:
    if opts.net_arch != 'orig' and opts.load_from_png:
        error('Not possible to load images with png format from a dataset different from EMOTIC.')
    # It is not  possible to load the weights from Torch, and at the same time restore a saved model:
    if opts.restore_model and opts.load_torch:
        error('Not possible to restore a saved model and at the same time load the weights from Torch.')
    # If the network architecture is set to one path, ensure it has its corresponding loss:
    if opts.net_arch == 'fullpath' and opts.loss_type != 'fullpath':
        error('fullpath network architecture must have fullpath loss_type')
    if opts.net_arch == 'bodypath' and opts.loss_type != 'bodypath':
        error('bodypath network architecture must have bodypath loss_type')


###########################################################################################################
### Preprocess image.
def preprocess_emotic(image, body_bbox, opts):
    # We assume the image comes with 3 dimensions, being [height, width, nchannels].
    # Also, the image is a PIL image.
    # body_bbox should have be a list with [x1, y1, x2, y2], the pixels of the corners of the
    # bounding box for the body.
    
    iW = image.size[0]
    iH = image.size[1]
    
    # Check if it is a black & white image:
    # (and put the image in "image_full")
    if image.mode == 'L': # black & white, 8-bit pixels
        image_full = image.convert('RGB')
    elif image.mode == 'RGB': # 3x8-bit pixels, true color
        image_full = image
    else:
        error('Unrecognized image mode %s' % image.mode)
    
    # Bounding box coordinates:
    x1 = np.max([body_bbox[0], 1])
    y1 = np.max([body_bbox[1], 1])
    x2 = np.min([body_bbox[2], iW])
    y2 = np.min([body_bbox[3], iH])
    
    # Crop the image containing the body:
    image_body = image_full.crop((x1-1, y1-1, x2, y2))
    
    # Resize:
    image_body = image_body.resize((128, 128), resample=Image.BILINEAR)
    
    # Resize the full image to ensure its lowest dimension is 256:
    if iW < iH:
        cW = 256
        cH = 256 * iH / iW
    else:
        cW = 256 * iW / iH
        cH = 256
    image_full = image_full.resize((cW, cH), resample=Image.BILINEAR)
    
    if opts.randomize_preprocess:
        h1 = np.ceil(np.random.rand() * (cH - 224))
        w1 = np.ceil(np.random.rand() * (cW - 224))
    else:
        h1 = np.ceil(0.5 * (cH - 224))
        w1 = np.ceil(0.5 * (cW - 224))
    
    # Crop the full image to be INPUT_LARGEST_SIZE on both width and height:
    image_full = image_full.crop((w1-1, h1-1, w1+224-1, h1+224-1))
    
    # Rescale to the range 0-1, and convert to floating point:
    image_full = np.float32(image_full) / 255
    image_body = np.float32(image_body) / 255
    
    return image_full, image_body


###########################################################################################################
### Preprocess image.
def preprocess_onepath(image, opts):
    # Decide image size:
    if opts.net_arch == 'fullpath':
        im_size = 224
    elif opts.net_arch == 'bodypath':
        im_size = 128
    else:
        error('net_arch option nor recognized.')
    # Resize:
    image_prep = image.resize((im_size, im_size), resample=Image.BILINEAR)
    # Rescale to the range 0-1, and convert to floating point:
    image_prep = np.float32(image_prep) / 255
    return image_prep





