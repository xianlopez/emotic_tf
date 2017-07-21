### General options for the system.
### This file is for defining the options that are available, and specifying default values.
### It is not intended to specify the options of a particular execution. That should be done
### in file options.py.
### Xian Lopez Alvarez
### 18/7/2017

class opts_cnn_emotic_1_class:
    def __init__(self):
        self.load_torch = False
        self.dirmodel = []
        self.correct_block2 = False
        self.correct_avgpool = False
        self.ldisc_c = 1.2 / 10
    

class general_options_class:
    def __init__(self):
        # Main actions:
        self.train = True
        self.evaluate_model = True
        
        # Specific CNN options:
        self.modelname = 'cnn_emotic_1'
        self.cnn_opts = {
            'cnn_emotic_1': opts_cnn_emotic_1_class()
        }
    
        # Training options:
        self.nsteps = 10000 # Number of training steps
        self.nsteps_print_batch_id = 1
        self.nsteps_trainloss = 200
        self.nsteps_valloss = 200
        self.nsteps_save = 1000
        self.initial_learning_rate = 1e-4
        
        self.momentum = 0.9
        
        # To restore a previously saved model:
        self.restore_model = False
        self.checkpoint = 'last' # 'last', or global_step to specify one.
        self.dir_saved_model = []
        
        # Memory limit for TensorFlow (fraction of total GPU's memory):
        self.memory_limit = 0.7
        
        # Other options:
        self.random_seed = -1 # A value lower than 0 means the seed is not set.
        self.randomize_preprocess = 1 # Take random cropping, or take 0.5 instead.
        self.normalize = 1 # 0: no division by std nor mean subtracted; 1: division by std and mean subtracted; 2: only mean subtracted
        self.shuffle = True # shuffle the dataset to change its order
        self.batch_size = 52
        self.dirbase = '/home/xian/eclipse-workspace/emotic_tf/'
#         self.seed = -1 # random seed (-1 means no seed specified)
        self.all_classes_in_batch = False
        self.keep_prob_train = 0.5 # for the dropout layer, during training.
        self.plot_training_history = False
        self.supress_random = False # Supress all random effects. This options will alter the value of other options.
        
        self.save_initial = False # Save the initial model, before training
        self.evaluate_initial = False # Evaluate on the validation set the initial model, before training.
        
        self.debug = False # Check if there are NaNs in variables and gradients during training
        self.nsteps_debug = 1
        
        self.loss_type = 'orig' # Select among the following options ('orig', 'onlycont', 'onlydisc', 'simple1'. 'fullpath'. 'bodypath')
        self.optimizer = 'adam' # Select among the following options ('amad', 'sgd', 'momentum', 'rmsprop')
        
        self.show_training_history = False
        self.nsteps_plotresults = 500
        
        self.net_arch = 'orig' # Network architecture. 'orig' for the original network, 'onlyfull' to have only the full image path, 'onlybody' to have only the boyd path.
        
        self.xavier_init = True # Use or not Xavier initialization.
        
        # Directories of databases:
        self.emotic_dir_jpg = '/home/xian/EMOTIC/EmpathyDB_images'
        self.emotic_dir_png = '/home/xian/EMOTIC/images_preprocessed'
        self.imagenet_dir = ''
        self.places_dir = ''
        
        self.load_from_png = False # Only with EMOTIC dataset. Load images from png format, which have already been preprocessed.
        self.mini_dataset = False # For debbugin. Take only a fraction of the original datasets, to speed up computations.
        self.mini_percent = 5 # Percentage to take when mini_dataset option is set to True.
        
        
        
        
        


