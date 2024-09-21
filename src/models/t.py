import torch
import torch.nn as nn

# 定义一个简单的单层神经网络
class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    def a(self):
        print('aaa')
    def forward(self, x):
        self.a()
        return self.linear(x)

# 创建一个模型实例，输入维度为 3，输出维度为 2
model = SimpleNet(input_size=3, output_size=2)

# 创建一个输入数据
input_data = torch.randn(1, 3)

# 执行前向传播
output = model(input_data)

# 打印输出结果
print(output)