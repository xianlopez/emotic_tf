### Main program. To be executed.
### Xian Lopez Alvarez
### 19/7/2017

import cnn_actions
import model_builder

# Load options:
from options import opts

# Create network and loss:
model_builder.build_model(opts)

if opts.train:
    # Train network:
    cnn_actions.train(opts)
   
if opts.evaluate_model:
    # Do evaluation:
    cnn_actions.evaluate_model(opts)