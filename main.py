### Main program. To be executed.
### Xian Lopez Alvarez
### 19/7/2017

import cnn_actions
import model_builder
import tools
import logging

# Load options:
from options import opts
    
# Create directory to store results:
dircase = tools.prepare_dircase(opts)

# Write options to text file:
tools.write_options(opts, dircase)

# Configure the logger:
tools.configure_logging(dircase)

# Create network and loss:
model_builder.build_model(opts)

# Train network:
if opts.train:
    cnn_actions.train(opts, dircase)
   
# Do evaluation:
if opts.evaluate_model:
    cnn_actions.evaluate_model(opts)
    
logging.info('Processed finished')