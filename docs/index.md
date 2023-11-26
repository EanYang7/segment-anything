# Segment Anything

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

![SAM design](assets/model_diagram.png?raw=true)

**Segment Anything Model (SAM)** 从诸如点或框之类的输入提示生成高质量的对象掩码，可用于为图像中的所有对象生成掩码。它已经在包含1100万张图像和11亿个掩码的[数据集](https://segment-anything.com/dataset/index.html)上进行了训练，并在各种分割任务上具有强大的零样本zero-shot性能。

<p float="left">
  <img src="assets/masks1.png?raw=true" width="37.25%" />
  <img src="assets/masks2.jpg?raw=true" width="61.5%" /> 
</p>



## 安装

代码要求`python>=3.8`，以及`pytorch>=1.7`和`torchvision>=0.8`。请按照[这里](https://pytorch.org/get-started/locally/)的说明安装PyTorch和TorchVision的依赖项。强烈建议安装支持CUDA的PyTorch和TorchVision。

---

克隆存储库并在本地安装：

```
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

> 建议先fork到自己仓库后再克隆

---

以下是必要的可选依赖项，用于掩码后处理、以COCO格式保存掩码、示例jupyter笔记本以及将模型导出为ONNX格式。运行示例jupyter笔记本还需要`jupyter`。

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## 入门

首先下载一个[模型检查点](#模型检查点)。然后，可以使用以下几行代码从给定提示中获取掩码：

```
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

或者为整个图像生成掩码：

```
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```

此外，可以使用命令行为图像生成掩码：

```
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```

有关更多详细信息，请参阅[使用提示生成掩码](https://eanyang7.github.io/segment-anything/notebooks/predictor_example/)和[自动生成对象掩码](https://eanyang7.github.io/segment-anything/notebooks/automatic_mask_generator_example/)的示例笔记本。

<p float="left">
  <img src="assets/notebook1.png?raw=true" width="49.1%" />
  <img src="assets/notebook2.png?raw=true" width="48.9%" />
</p>



## ONNX导出

SAM的轻量级掩码解码器可以导出为ONNX格式，以便在支持ONNX运行时的任何环境中运行，例如在[演示](https://segment-anything.com/demo)中展示的浏览器中。使用以下命令导出模型：

```
python scripts/export_onnx_model.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

请参阅[示例笔记本](https://eanyang7.github.io/segment-anything/notebooks/onnx_model_example/)以了解如何通过SAM的骨干进行图像预处理，然后使用ONNX模型进行掩码预测的详细信息。建议使用PyTorch的最新稳定版本进行ONNX导出。

### Web演示

`demo/`文件夹中有一个简单的单页React应用程序，展示了如何在支持多线程的Web浏览器中使用导出的ONNX模型运行掩码预测。请查看[`demo/README.md`](https://github.com/facebookresearch/segment-anything/blob/main/demo/README.md)以获取更多详细信息。

## 模型检查点

提供了三个模型版本，具有不同的骨干大小。可以通过运行以下代码实例化这些模型：

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

单击下面的链接下载相应模型类型的检查点。

- **`default`或`vit_h`：[ViT-H SAM模型。](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`：[ViT-L SAM模型。](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`：[ViT-B SAM模型。](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

> b:base基础模型
>
> l:large较大模型
>
> h:huge最大的模型

## 数据集

请参阅[此处](https://ai.facebook.com/datasets/segment-anything/)以获取有关数据集的概述。可以在[此处](https://ai.facebook.com/datasets/segment-anything-downloads/)下载数据集。通过下载数据集，您同意已阅读并接受了SA-1B数据集研究许可条款。

每个图像的掩码保存为json文件。它可以在以下格式的Python字典中加载。

```python
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "image_id"              : int,              # 图像id
    "width"                 : int,              # 图像宽度
    "height"                : int,              # 图像高度
    "file_name"             : str,              # 图像文件名
}

annotation {
    "id"                    : int,              # 注释id
    "segmentation"          : dict,             # 以COCO RLE格式保存的掩码。
    "bbox"                  : [x, y, w, h],     # 掩码周围的框，以XYWH格式表示
    "area"                  : int,              # 掩码的像素面积
    "predicted_iou"         : float,            # 模型对掩码质量的自身预测
    "stability_score"       : float,            # 掩码质量的度量
    "crop_box"              : [x, y, w, h],     # 用于生成掩码的图像的裁剪，以XYWH格式表示
    "point_coords"          : [[x, y]],         # 输入模型生成掩码的点坐标
}
```

图像ID可以在`sa_images_ids.txt`中找到，可以使用上述[链接](https://ai.facebook.com/datasets/segment-anything-downloads/)下载。

要将COCO RLE格式的掩码解码为二进制：

```
from pycocotools import mask as mask_utils
mask = mask_utils.decode(annotation["segmentation"])
```

请参阅[此处](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py)以获取有关如何操作以RLE格式存储的掩码的更多说明。

## 引用Segment Anything

如果您在研究中使用SAM或SA-1B，请使用以下BibTeX条目。

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```