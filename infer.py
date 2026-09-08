import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2  # 导入OpenCV库用于图像处理

# 定义模型
class AreaModel(nn.Module):
    def __init__(self):
        super(AreaModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
def load_model(filepath):
    model = AreaModel()
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

# 预测图像面积
def predict_area(model, image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 将图像转换为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用阈值化来分割图像
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # 找到轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 假设最大的轮廓是我们要计算面积的形状
    largest_contour = max(contours, key=cv2.contourArea)
    # 计算轮廓的边界框
    x, y, w, h = cv2.boundingRect(largest_contour)
    # 获取形状的尺寸
    shape_size = np.array([w, h], dtype=np.float32)
    # 将尺寸转换为模型输入格式
    shape_size = torch.tensor(shape_size).unsqueeze(0)
    # 预测面积
    with torch.no_grad():
        predicted_area = model(shape_size)
    return predicted_area.item()

# 主程序
if __name__ == "__main__":
    # 加载模型
    model = load_model("model.pth")
    
    # 预测图像面积
    image_path = "3.png"  # 替换为你的图像路径
    predicted_area = predict_area(model, image_path)
    print(f"Predicted area of the shape in the image: {predicted_area}")