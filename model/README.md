# ShanShuiSeg-B5

中国山水画 14 类语义分割模型，供非商业科研和评估使用。安装依赖后即可推理，无需训练数据、预训练权重或原工程。

A 14-class semantic segmentation model for Chinese landscape paintings, for non-commercial research and evaluation. Inference runs from this folder with the supplied checkpoint.

## 权重 / Checkpoint

将 `shanshuiseg_b5.pth` 放在本目录中；完整模型包已包含此文件。若使用 GitHub 源码包，请从仓库 Releases 获取权重，或使用包含权重的完整模型包。

Place `shanshuiseg_b5.pth` in this folder. It is included in the complete model package. For a source-only checkout, obtain the checkpoint from the repository's Releases or use the complete package.

- [下载模型权重 / Download checkpoint](https://github.com/shengbei4980/ShanShuiSeg/releases/download/v1.0.0/shanshuiseg_b5.pth)
- [下载完整模型包 / Download complete package](https://github.com/shengbei4980/ShanShuiSeg/releases/download/v1.0.0/ShanShuiSeg-v1.0.0.zip)

## 安装 / Install

使用 Windows/Linux x86-64、Python 3.8.20、PyTorch 1.12.1、MMCV-full 1.6.2 和 MMSegmentation 0.30.0。以下命令在 `model` 目录中运行，CPU 和 CUDA 路线二选一。

Use Python 3.8.20 on Windows/Linux x86-64. Run these commands inside `model`, choosing either CPU or CUDA.

```bash
conda create -n shanshuiseg python=3.8.20 -y
conda activate shanshuiseg
```

**CPU**

```bash
python -m pip install torch==1.12.1+cpu torchvision==0.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install mmcv-full==1.6.2 --only-binary=mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.12.0/index.html
```

**NVIDIA GPU / CUDA 11.3**

```bash
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install mmcv-full==1.6.2 --only-binary=mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
```

然后安装其余依赖 / Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

MMCV 的 `torch1.12.0` 安装包用于 PyTorch 1.12.1；本包使用上述固定版本。安装方式见 [PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)、[MMCV 1.6.2 安装说明](https://mmcv.readthedocs.io/en/v1.6.2/get_started/installation.html)和 [MMSegmentation 0.30 兼容说明](https://mmsegmentation.readthedocs.io/en/0.x/faq.html)。

The MMCV `torch1.12.0` wheels support PyTorch 1.12.1. Keep the versions above; see the linked official installation guides.

## 推理 / Inference

单图 / Single image:

```bash
python infer.py --input "painting.jpg" --output "results"
```

目录（含子目录）/ Directory, including subdirectories:

```bash
python infer.py --input "paintings" --output "results" --device cuda:0 --mode sliding --batch-size 1
```

CPU 与整图缩放 / CPU and whole-image resizing:

```bash
python infer.py --input "painting.jpg" --output "results_resize" --checkpoint "shanshuiseg_b5.pth" --device cpu --mode resize
```

| 参数 / Option | 用法 / Meaning |
| --- | --- |
| `--input` | 必填：单图或目录 / Required: image or directory |
| `--output` | 必填：结果目录 / Required: output directory |
| `--checkpoint` | 默认本目录的 `shanshuiseg_b5.pth` / Defaults to the checkpoint beside the script |
| `--device` | 默认自动选择 GPU/CPU；可设 `cpu` 或 `cuda:0` / Automatically selects GPU/CPU; accepts `cpu` or `cuda:0` |
| `--mode` | 默认 `sliding`；可选 `resize` / Default: `sliding`; alternative: `resize` |
| `--batch-size` | 默认 `1`；`sliding` 模式每批切片数 / Default: `1`; tiles per batch in `sliding` mode |

`sliding` 在原图上按 512 × 512 切片、384 像素步长推理，融合重叠区域。`resize` 将整图缩放至 512 × 512 推理。两种模式均输出原图尺寸的结果；增加滑窗批量会增加显存或内存占用。

`sliding` uses 512 × 512 tiles with a 384-pixel stride and merges overlapping predictions. `resize` processes the whole image at 512 × 512. Both modes return results at the original image size. Larger sliding-window batches require more GPU or system memory.

## 输出 / Outputs

每张输入图生成以下结果，目录输入会保留相对目录结构；文件名保留原扩展名以区分同名图片。例如 `painting.jpg` 对应：

Each input produces the following files. Relative subdirectories and the original filename extension are retained:

```text
results/
  labels/painting.jpg_labels.png
  color/painting.jpg_color.png
  overlay/painting.jpg_overlay.jpg
```

`labels` 是单通道类别图，像素值为 **1–14**；`color` 是彩色类别图，`overlay` 是原图叠加图。类别名称、ID 和颜色见 [labels.json](labels.json)。原始标签值 0 不属于这 14 个类别，模型不会输出 0 类。

`labels` contains single-channel class IDs **1–14**; `color` contains the color mask; `overlay` contains the blended image. Names, IDs and colors are in [labels.json](labels.json). Original label ID 0 is outside the model's 14 classes.

输入和输出目录应分开；已有结果不会被覆盖，请使用新的输出目录。

Keep input and output directories separate. Existing results are not overwritten; choose a new output directory for another run.

## 许可 / License

模型及本包推理代码采用随包 [LICENSE](LICENSE) 中的 NVIDIA SegFormer 许可，仅限非商业研究或评估。MMSegmentation 的 Apache 2.0 声明同样保留在该文件中。

The model and package-specific inference code are subject to the NVIDIA SegFormer license in [LICENSE](LICENSE), restricting use to non-commercial research or evaluation. The same file retains MMSegmentation's Apache 2.0 notices.
