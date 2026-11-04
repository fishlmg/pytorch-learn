import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 定义数据集类
class ShapeDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.shapes = torch.randn(num_samples, 2)  # 随机生成形状的参数
        self.areas = self.shapes[:, 0] * self.shapes[:, 1]  # 计算面积

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, idx):
        return self.shapes[idx], self.areas[idx]

# 定义模型
class AreaModel(nn.Module):
    def __init__(self):
        super(AreaModel, self).__init__()
        self.fc1 = nn.Linear(2, 128)  # 增加神经元数量
        self.fc2 = nn.Linear(128, 256)  # 增加神经元数量并增加一个隐藏层
        self.fc3 = nn.Linear(256, 128)  # 增加神经元数量
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 训练模型
def train_model(model, criterion, optimizer, dataloader, device, epochs=2000):
    model.to(device)  # 将模型移动到 GPU
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.unsqueeze(1).to(device)  # 将数据移动到 GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:  # 每100轮打印一次损失
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 测试模型
def test_model(model, test_inputs, device):
    model.eval()
    test_inputs = test_inputs.to(device)  # 将数据移动到 GPU
    with torch.no_grad():
        test_outputs = model(test_inputs)
    print(f'Predicted areas: {test_outputs.cpu().numpy()}')  # 将结果移回 CPU 并转换为 NumPy

# 保存模型
def save_model(model, filename="model1.pth"):
    torch.save(model.state_dict(), filename)

# 主程序
if __name__ == "__main__":
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 生成数据
    num_samples = 1000
    dataset = ShapeDataset(num_samples)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化模型
    model = AreaModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, criterion, optimizer, dataloader, device, epochs=800)

    # 测试模型
    test_inputs = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    test_model(model, test_inputs, device)

    # 保存模型
    save_model(model)