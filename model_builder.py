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
    cnn_builder.define_loss()
    
    # Define optimizer:
    cnn_builder.define_optimizer(opts)
