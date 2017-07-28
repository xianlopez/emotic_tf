### File for specifying the options that will be executed.
### Xian Lopez Alvarez
### 18/7/2017

from options_def import general_options_class

opts = general_options_class()

# Add below the options you want to apply.
# Missing options here will take default values (the ones specified in options_def.py).

opts.modelname = 'cnn_emotic_2'

opts.normalize = 0
opts.shuffle = 1

opts.batch_size = 64

opts.supress_random = 1

# opts.nsteps = 4000
opts.nsteps = 3000
opts.nsteps_print_batch_id = 1
opts.nsteps_trainloss = 5000
opts.nsteps_valloss = 5000
opts.nsteps_save = 200
opts.nsteps_plotresults = 5000
opts.initial_learning_rate = 1e-4
        
opts.momentum = 0.5

opts.net_arch = 'bodypath'

opts.memory_limit = 1

opts.train = 1
opts.evaluate_model = 0

opts.evaluate_initial = 0
opts.save_initial = 0

opts.debug = 1
opts.nsteps_debug = 1

opts.loss_type = 'bodypath'
opts.optimizer = 'momentum'

opts.xavier_init = 0
opts.load_from_png = 0

opts.restore_model = 0
opts.checkpoint = 'last'
opts.dir_saved_model = '/home/xian/eclipse-workspace/emotic_tf/experiments/day_2017_07_20/case_145'

# CNN EMOTIC options:
opts.cnn_opts['cnn_emotic_1'].load_torch = False
opts.cnn_opts['cnn_emotic_1'].dirmodel = opts.dirbase + 'models/model_15_dropbox/'
opts.cnn_opts['cnn_emotic_1'].correct_block2 = True
opts.cnn_opts['cnn_emotic_1'].correct_avgpool = True

opts.mini_dataset = 1
opts.mini_percent = 1

opts.w_cont = 1
opts.w_disc = 1./6.
opts.loss_cont_margin = 0.1
opts.loss_cont_saturation = 1500
opts.reweight = 'allones'






