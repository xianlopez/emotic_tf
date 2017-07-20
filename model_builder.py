### Import all the defined models, and build the one that is specified.
### Xian Lopez Alvarez
### 18/7/2017

import tools

import cnn_emotic_1




def build_model(opts):
    
    if opts.modelname == 'cnn_emotic_1':
        cnn_builder = cnn_emotic_1.cnn_builder_class(opts.cnn_opts[opts.modelname], opts.batch_size)
        
    else:
        tools.error('modelname not recognized.')

    # Define network:
    cnn_builder.define_network()
    
    # Define loss:
    if opts.loss_type == 'orig':
        cnn_builder.define_loss_orig()
    elif opts.loss_type == 'onlycont':
        cnn_builder.define_loss_onlycont()
    elif opts.loss_type == 'onlydisc':
        cnn_builder.define_loss_onlydisc()
    elif opts.loss_type == 'simple1':
        cnn_builder.define_loss_simple1()
    else:
        tools.error('Loss type not recognized.')
    
    # Define optimizer:
    gradients = cnn_builder.define_optimizer(opts)
    
    return gradients
