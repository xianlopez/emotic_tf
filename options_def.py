### General options for the system.
### This file is for defining the options that are available, and specifying default values.
### It is not intended to specify the options of a particular execution. That should be done
### in file options.py.
### Xian Lopez Alvarez
### 18/7/2017

class opts_cnn_emotic_1_class:
    load_torch = False
    dirmodel = []
    correct_block2 = False
    correct_avgpool = False
    

class general_options_class:
    # Main actions:
    train = True
    evaluate_model = True
    
    modelname = 'cnn_emotic_1'
    # Other options:
    normalize = 1 # 0: no division by std nor mean subtracted; 1: division by std and mean subtracted; 2: only mean subtracted
    shuffle = False # shuffle the dataset to change its order
    batch_size = 2
    dirbase = '/home/xian/eclipse-workspace/emotic_tf/'
    seed = -1 # random seed (-1 means no seed specified)
    all_classes_in_batch = False
    keep_prob_train = 0.5 # for the dropout layer, during training.
    
    cnn_opts = {
        'cnn_emotic_1': opts_cnn_emotic_1_class()
    }


    # Training options:
    nsteps = 10000 # Number of training steps
    nsteps_print_batch_id = 1
    nsteps_trainloss = 200
    nsteps_valloss = 200
    nsteps_save = 1000
    initial_learning_rate = 1e-4