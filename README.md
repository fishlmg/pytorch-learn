# PyTorch 学习练习

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

用面积回归这个小例子，把自定义网络、`nn.Sequential`、训练循环和加载权重做推理串起来。训练数据是随机宽高，标签是宽乘高；预测时从图片轮廓里取出宽高。

## 📖 适合谁

- 刚学完 Python、想入门深度学习的同学
- 想通过最小可运行例子理解 `nn.Module` / `nn.Sequential` / `DataLoader` 的使用者

## 🗂 内容覆盖

| 文件 | 内容 |
| --- | --- |
| `nn_base.ipynb` | `nn.Module` 自定义网络的最小模板 |
| `sequential.ipynb` | 用 `nn.Sequential` 堆一个面积回归网络 |
| `train.ipynb` | Notebook 里的训练循环（Dataset → DataLoader → Loss） |
| `train_area.py` | 脚本版训练：随机宽高回归面积，抽查两组尺寸，保存 `model1.pth` |
| `predict_area.py` | 读图片轮廓的宽高，用 `model1.pth` 预测面积 |
| `infer.py` | 另一版推理：较浅的两层网络，加载 `model.pth` |

## 🚀 快速开始

```bash
pip install -r requirements.txt
```

先看 Notebook，按顺序打开：

```bash
jupyter notebook
```

1. `nn_base.ipynb` — 网络类怎么写、前向传播从哪走
2. `sequential.ipynb` — 用 Sequential 把线性层叠起来
3. `train.ipynb` — 在 Notebook 里把训练循环跑通（只打印 loss，不保存权重）

再跑脚本。`train_area.py` 不读外部数据，自己造 1000 组宽高，训完在当前目录写出 `model1.pth`，并打印 `(3, 4)`、`(5, 6)` 的预测面积：

```bash
python train_area.py
```

`predict_area.py` 加载这份 `model1.pth`。默认读当前目录的 `3.png`（画面里一块形状即可），路径写在脚本的 `image_path`，换成自己的图再运行：

```bash
python predict_area.py
```

`infer.py` 是更早的一版，网络只有两层，读的是 `model.pth`。仓库里没有这份权重，也不能拿 `model1.pth` 去替，层数对不上。

## 📦 依赖

PyTorch、torchvision、NumPy、Matplotlib、OpenCV —— 详见 [requirements.txt](requirements.txt)

## 📝 说明

训练产出的权重（`.pth`）和结果图片（`.png`）不入库，见 `.gitignore`。`train.ipynb` 只用来看 loss；要拿到能给 `predict_area.py` 用的权重，跑 `train_area.py`。

图片预测假定最大的那块轮廓就是要测的形状，取它的外接矩形宽高送进网络。

## 📄 许可证

[MIT License](LICENSE)
