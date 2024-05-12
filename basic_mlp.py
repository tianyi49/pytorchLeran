import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

# 定义输入、隐藏层和输出层的大小
input_size = 3
hidden_size = 6
output_size = 2  # 二分类，输出大小为2

# 创建MLP模型实例
mlp = MLP(input_size, hidden_size, output_size)

# 示例输入
input_data = torch.tensor([[0.1, 0.2, 0.3]])  # 示例输入数据，大小为(batch_size, input_size)

# 使用模型进行前向传播
output = mlp(input_data)

print("Output probabilities:", output)