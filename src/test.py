import torch

if __name__ == '__main__':
    loaded_a = torch.load('my_tensor.pt')

    # Print the loaded tensor
    print(loaded_a)