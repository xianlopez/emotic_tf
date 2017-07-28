### Import all the defined models, and build the one that is specified.
### Xian Lopez Alvarez
### 18/7/2017

import tools

import cnn_emotic_1
import cnn_emotic_2




def build_model(opts):
    
    if opts.modelname == 'cnn_emotic_1':
        cnn_builder = cnn_emotic_1.cnn_builder_class(opts.cnn_opts[opts.modelname], opts)
    
    elif opts.modelname == 'cnn_emotic_2':
        cnn_builder = cnn_emotic_2.cnn_builder_class()
        
    else:
        tools.error('<model_builder> modelname not recognized.')

    # Define network:
    if opts.net_arch == 'orig':
        cnn_builder.define_network()
    elif opts.net_arch == 'fullpath':
        cnn_builder.define_fullpath()
    elif opts.net_arch == 'bodypath':
        cnn_builder.define_bodypath()
    else:
        tools.error('<model_builder> Network architecture not recognized.')
    
    # Define loss:
    if opts.loss_type == 'orig' or opts.loss_type == 'onlycont' or opts.loss_type == 'onlydisc':
        cnn_builder.define_loss_orig(opts)
    elif opts.loss_type == 'simple1':
        cnn_builder.define_loss_simple1()
    elif opts.loss_type == 'fullpath':
        cnn_builder.define_loss_fullpath()
    elif opts.loss_type == 'bodypath':
        cnn_builder.define_loss_bodypath(opts)
    else:
        tools.error('<model_builder> Loss type not recognized.')
    
    # Define optimizer:
    gradients = cnn_builder.define_optimizer(opts)
    
    return gradients
