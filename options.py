### File for specifying the options that will be executed.
### Xian Lopez Alvarez
### 18/7/2017

from options_def import general_options_class

opts = general_options_class()

# Add below the options you want to apply.
# Missing options here will take default values (the ones specified in options_def.py).

opts.normalize = 2
opts.shuffle = 1

opts.batch_size = 100

opts.supress_random = 1

# opts.nsteps = 4000
opts.nsteps = 1
opts.nsteps_print_batch_id = 10
opts.nsteps_trainloss = 200
opts.nsteps_valloss = 200
opts.nsteps_save = 2000
opts.initial_learning_rate = 1e-6

opts.net_arch = 'orig'

opts.train = 0
opts.evaluate_model = 1

opts.evaluate_initial = 0
opts.save_initial = 0

opts.debug = 1
opts.nsteps_debug = 50

opts.loss_type = 'orig'
opts.optimizer = 'momentum'

opts.xavier_init = 1
opts.load_from_png = 1

opts.restore_model = 0
opts.checkpoint = 'last'
opts.dir_saved_model = '/home/xian/eclipse-workspace/emotic_tf/experiments/day_2017_07_20/case_145'

# CNN EMOTIC options:
opts.cnn_opts['cnn_emotic_1'].load_torch = True
opts.cnn_opts['cnn_emotic_1'].dirmodel = opts.dirbase + 'models/model_15_dropbox/'
opts.cnn_opts['cnn_emotic_1'].correct_block2 = False
opts.cnn_opts['cnn_emotic_1'].correct_avgpool = False

opts.mini_dataset = 0
opts.mini_percent = 10

opts.w_cont = 1
opts.w_disc = 1./6.
opts.loss_cont_margin = 0.1
opts.loss_cont_saturation = 1500
opts.reweight = 'ondataset'






