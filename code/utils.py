import argparse
import importlib
import sys

def import_model():
    # Import Correct Model
    parser = argparse.ArgumentParser(description='Run a specified model.')
    parser.add_argument('model_name', type=str, help='The name of the model to run')
    # Parse arguments
    args = parser.parse_args()
    # Prepend 'models.' to the model name
    full_module_name = f'models.{args.model_name}'

    # Dynamically import the specified model
    try:
        model = importlib.import_module(full_module_name)
        globals()[args.model_name] = model
        print(f'Successfully imported {full_module_name}')
    except ImportError:
        print(f'Error: Could not import module {full_module_name}')
        sys.exit(1)

    # Example usage of the imported model
    if hasattr(model, 'run'):
        model.run()
    else:
        print(f'Error: {args.model_name} does not have a run() function')
