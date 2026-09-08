# PyTorch 学习练习

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

PyTorch 深度学习入门到实践的练习集合，以可运行的 Notebook 和脚本覆盖张量操作、网络搭建、训练循环与模型推理全流程。

## 📖 适合谁

- 刚学完 Python、想入门深度学习的同学
- 想通过最小可运行例子理解 `nn.Module` / `nn.Sequential` / `DataLoader` 的使用者

## 🗂 内容覆盖

| 文件 | 内容 |
| --- | --- |
| `1.py` | PyTorch 基础语法练习（张量、自动求导） |
| `nn_base.ipynb` | `nn.Module` 自定义网络构建 |
| `sequential.ipynb` | `nn.Sequential` 快速堆叠网络层 |
| `train.ipynb` | 完整训练流程（Dataset → DataLoader → 训练循环 → Loss 曲线） |
| `test.py` | 模型评估 |
| `infer.py` | 加载权重进行推理 |

## 🚀 快速开始

```bash
pip install -r requirements.txt
jupyter notebook
```

按顺序打开学习：

1. `nn_base.ipynb` — 理解网络层与前向传播
2. `sequential.ipynb` — 快速搭建网络
3. `train.ipynb` — 跑通完整训练循环

## 📦 依赖

PyTorch、torchvision、NumPy、Matplotlib、OpenCV —— 详见 [requirements.txt](requirements.txt)

## 📝 说明

训练产出的权重文件（`.pth`）与结果图片（`.png`）不包含在仓库中（`.gitignore` 已排除），运行 Notebook 后会自动生成。

## 📄 许可证

[MIT License](LICENSE)
