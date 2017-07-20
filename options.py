### File for specifying the options that will be executed.
### Xian Lopez Alvarez
### 18/7/2017

from options_def import general_options_class

opts = general_options_class()

# Add below the options you want to apply.
# Missing options here will take default values (the ones specified in options_def.py).

opts.normalize = 2
opts.shuffle = 1

opts.batch_size = 64

opts.supress_random = 1

opts.nsteps = 2000
opts.nsteps_print_batch_id = 1
opts.nsteps_trainloss = 100
opts.nsteps_valloss = 100
opts.nsteps_save = 1000
opts.initial_learning_rate = 1e-5

opts.train = 1
opts.evaluate_model = 0

opts.evaluate_initial = 0
opts.save_initial = 0

opts.debug = 1

opts.restore_model = 0
opts.checkpoint = 'last'
opts.dir_saved_model = '/home/xian/eclipse-workspace/emotic_tf/experiments/day_2017_07_20/case_145'

# CNN EMOTIC options:
opts.cnn_opts['cnn_emotic_1'].load_torch = True
opts.cnn_opts['cnn_emotic_1'].dirmodel = opts.dirbase + 'models/model_15_dropbox/'
opts.cnn_opts['cnn_emotic_1'].correct_block2 = False
opts.cnn_opts['cnn_emotic_1'].correct_avgpool = False

