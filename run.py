import sys
import os
import argparse
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'code')))
from code.data_loading import load_all_data
from code.utils import import_model
import code.train as train

## Dynamically Import Correct Model
parser = argparse.ArgumentParser(description='Run a specified model.') # Set up argument parser
parser.add_argument('model_name', type=str, help='The name of the model to run')
args = parser.parse_args() # Parse arguments
full_module_name = f'code.models.{args.model_name}'
try:
    module = importlib.import_module(full_module_name)
    ModelClass = getattr(module, args.model_name.capitalize())
    globals()[args.model_name] = ModelClass
except ImportError:
    print(f'Error: Could not import module {full_module_name}')
    sys.exit(1)
except AttributeError:
    print(f'Error: Module {full_module_name} does not have a class named {args.model_name}')
    sys.exit(1)

# create instance of model
model = ModelClass()

##  Train Model
try:
    print(f'Training Model...')
    train.train_model(model)
except Exception as e:
    print(f'An unexpected error occurred: {e}')
    sys.exit(1)

##  Evaluate Model
try:
    print(f'Evaluating Model...')
    train.evaluate_model(model)
except Exception as e:
    print(f'An unexpected error occurred: {e}')
    sys.exit(1)
