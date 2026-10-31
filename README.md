# ShanShuiSeg

**ShanShuiSeg-B5：中国山水画 14 类语义分割模型**

ShanShuiSeg-B5 基于 SegFormer-B5，面向中国山水画中的景观要素识别与像素级分割。本仓库提供训练好的模型权重、独立推理脚本及使用说明，便于研究者对自己的山水画图像进行分割。

A SegFormer-B5-based model for 14-class semantic segmentation of Chinese landscape paintings. This repository provides trained weights and standalone inference code for research and evaluation.

## 下载 / Download

**推荐下载 [完整模型包（约 290 MiB）](https://github.com/shengbei4980/ShanShuiSeg/releases/download/v1.0.0/ShanShuiSeg-v1.0.0.zip)**。解压后只有一个 `model` 文件夹，包含模型权重和运行所需代码。

- [完整模型包 / Complete package](https://github.com/shengbei4980/ShanShuiSeg/releases/download/v1.0.0/ShanShuiSeg-v1.0.0.zip)
- [单独下载权重 / Checkpoint only（约 313 MiB）](https://github.com/shengbei4980/ShanShuiSeg/releases/download/v1.0.0/shanshuiseg_b5.pth)
- [版本发布页 / Release v1.0.0](https://github.com/shengbei4980/ShanShuiSeg/releases/tag/v1.0.0)

使用 Git clone 或 GitHub 的 **Code → Download ZIP** 仅能获得代码；请另外下载权重并放入 `model/`。

The complete package includes the checkpoint. A Git clone or GitHub source archive contains code only; download the checkpoint separately into `model/`.

## 快速使用 / Quick start

1. 下载并解压完整模型包。
2. 按照 [安装说明](model/README.md#安装--install) 创建环境并安装依赖，选择 CPU 或 NVIDIA GPU 版本。
3. 在 `model` 目录运行：

```bash
python infer.py --input "painting.jpg" --output "results"
```

批量处理目录及子目录：

```bash
python infer.py --input "paintings" --output "results" --device cuda:0 --mode sliding --batch-size 1
```

默认使用 512 × 512 滑窗、384 像素步长，在原图分辨率下融合预测。也可使用 `--mode resize` 进行整图缩放推理。输出包括原图尺寸的类别标签图、彩色分割图和叠加图。

Follow the [installation and usage guide](model/README.md), then run inference from `model/`. Both individual images and directories are supported. Outputs include class-ID masks, color masks and image overlays at the original resolution.

## 分割类别 / Classes

模型输出 14 个类别，标签图像素值为 **1–14**：

| ID | 类别 | Class |
| --- | --- | --- |
| 1 | 河流 | river |
| 2 | 山体 | mountain |
| 3 | 树木 | trees |
| 4 | 云 | clouds |
| 5 | 建筑 | building |
| 6 | 动物 | animals |
| 7 | 鸟 | birds |
| 8 | 舟船 | boat |
| 9 | 人物 | person |
| 10 | 天空 | sky |
| 11 | 桥梁 | bridge |
| 12 | 瀑布 | waterfall |
| 13 | 岩石 | rock |
| 14 | 草地 | grass |

类别顺序与颜色定义见 [labels.json](model/labels.json)。

## 文件说明 / Files

完整模型包的目录结构：

```text
model/
├── shanshuiseg_b5.pth   # 模型权重（从 Release 下载）
├── infer.py            # 单图和目录推理
├── model_config.py     # 模型结构配置
├── labels.json         # 类别 ID 与颜色
├── requirements.txt    # Python 依赖
├── README.md           # 详细安装与使用说明
└── LICENSE             # 许可与第三方声明
```

运行环境：Python 3.8.20、PyTorch 1.12.1、MMCV-full 1.6.2、MMSegmentation 0.30.0。完整安装命令与参数说明见 [model/README.md](model/README.md)。

## 许可 / License

模型及本包推理代码采用 [NVIDIA SegFormer 许可](model/LICENSE)，限非商业研究或评估使用。MMSegmentation 的 Apache 2.0 许可声明保留在同一文件中。

The model and package-specific inference code are provided for non-commercial research or evaluation under the NVIDIA SegFormer license. See [LICENSE](model/LICENSE) for the full terms and third-party notices.


## Supplementary data

The processed datasets for 5,915 Chinese landscape-painting images are available in [data/](data/). They include C1-C4 spatial-expression classifications, English visual evidence and classification justifications, and landscape-element segmentation counts and area fractions. See the [data description](data/README.md) for file contents and record linkage.
