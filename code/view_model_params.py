import torch

def view_model_params(model_name):
    state_dict = torch.load(f'model_info/{model_name}.pth')

    # Print the state dictionary keys
    print(state_dict.keys())
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")

if __name__ == '__main__':
    import sys
    model_name = sys.argv[1]
    view_model_params(model_name)
