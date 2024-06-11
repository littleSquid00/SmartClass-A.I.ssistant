import torch
import sys
sys.path.append('./models')
# for path in sys.path:
#     print(path)

# from models.model1 import Model1

# def view_model_params(model_name):
#     state_dict = torch.load(f'model_info/{model_name}/{model_name}.pth')
#     model = Model1()
#     model.load_state_dict(state_dict)
#    # Iterate over all weights and display them
#     for name, param in model.named_parameters():
#         print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}")

def view_model_performance(model_name):
    f = open(f'model_info/{model_name}/performance.txt', 'r')
    content = f.read()
    print(content)

# if __name__ == '__main__':
#     model_name = sys.argv[1]
#     print("Model Performance")
#     view_model_performance(model_name)
#     print("\nModel Weights")
#     view_model_params(model_name)
