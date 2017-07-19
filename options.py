### File for specifying the options that will be executed.
### Xian Lopez Alvarez
### 18/7/2017

from options_def import general_options_class

opts = general_options_class()

# Add below the options you want to apply.
# Missing options here will take default values (the ones specified in options_def.py).

opts.normalize = 2
opts.nsteps = 10

opts.train = 1
opts.evaluate_model = 0

# CNN EMOTIC options:
opts.cnn_opts['cnn_emotic_1'].load_torch = True
opts.cnn_opts['cnn_emotic_1'].dirmodel = opts.dirbase + 'models/model_15_dropbox/'
opts.cnn_opts['cnn_emotic_1'].correct_block2 = False
opts.cnn_opts['cnn_emotic_1'].correct_avgpool = False

