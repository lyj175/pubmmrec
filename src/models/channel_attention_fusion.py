import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
  def __init__(self, in_planes, ratio=2):
    super(ChannelAttention, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)

    self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
    self.fc3 = nn.Conv2d(in_planes, 64, 1, bias=False)

    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device) 
    self.to(device)
    # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
    avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
    max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
    out = avg_out + max_out
    result = self.fc3(out)
    return self.sigmoid(result)


# def example():
#     import torch
#     import torch.nn as nn
#     # Sample input tensor
#     x = torch.randn(1, 64, 50, 50)  # Batch size 1, 64 channels, 50x50 feature map
#     # (19000,10,1,60) shape
#
#     # Instantiate the ChannelAttention module
#     ca = ChannelAttention(in_planes=64)
#     # Apply channel attention to the input
#     output = ca(x)
#
#     # Print the output shape
#     print(output.shape)
#     print(output)
# example()