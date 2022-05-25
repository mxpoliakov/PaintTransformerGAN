# Paint Transformer: Feed Forward Neural Painting with Stroke Prediction

> [[Paper](https://arxiv.org/abs/2108.03798)] [[Official Paddle Implementation](https://github.com/wzmsltw/PaintTransformer)] [[Huggingface Gradio Demo](https://huggingface.co/spaces/akhaliq/PaintTransformer)] [[Unofficial PyTorch Re-Implementation](https://github.com/Huage001/PaintTransformer)] [[Colab](https://colab.research.google.com/drive/1m2gAYOdRIQVxrnVySmA-Pug0H_I13-Sp?usp=sharing)]

## Overview

This repository contains the officially **unofficial** PyTorch **re**-implementation of paper:

*Paint Transformer: Feed Forward Neural Painting with Stroke Prediction*,

Songhua Liu\*, Tianwei Lin\*, Dongliang He, Fu Li, Ruifeng Deng, Xin Li, Errui Ding, Hao Wang (* indicates equal contribution)

ICCV 2021 (Oral)

![](picture/picture.png)

## Prerequisites

* Linux or macOS
* Python 3
* PyTorch 1.7+ and other dependencies (torchvision and other common python libs)

## Results

<style>
table th:first-of-type {
    width: 33.33%;
}
table th:nth-of-type(2) {
    width: 33.33%;
}
table th:nth-of-type(3) {
    width: 33.33%;
}
</style>

| Image | PaintTransformer | PaintTransformerGAN |
|---|---|---|
| ![](picture/1.png) | ![](picture/1a.png) | ![](picture/1b.png) |
| ![](picture/2.png) | ![](picture/2a.png) | ![](picture/2b.png) |
| ![](picture/3.png) | ![](picture/3a.png) | ![](picture/3b.png) |
| ![](picture/4.png) | ![](picture/4a.png) | ![](picture/4b.png) |

## Getting Started

* Clone this repository:

  ```shell
  git clone https://github.com/Huage001/PaintTransformerGAN
  cd PaintTransformerGAN
  ```

* Inference: 

  ```shell
  python inference.py
  ```

  * Input image path, output path, and etc can be set in the main function.
  * Notably, there is a flag *serial* as one parameter of the main function:
    * If *serial* is True, strokes would be rendered serially. The consumption of video memory will be low but it requires more time.
    * If *serial* is False, strokes would be rendered in parallel. The consumption of video memory will be high but it would be faster.
    * If animated results are required, *serial* must be True.

* Train:

  * Before training, start tensorboard:

    ```shell
    tensorboard serve
    ```

  * Then, simply run: 

    ```shell
    cd train
    bash train.sh
    ```

  * You can monitor training status at http://localhost:6006/ and models would be saved at checkpoints/painter folder.

* You may feel free to try other training options written in train.sh. 

## Citation

* If you find ideas or codes useful for your research, please cite:

  ```
  @inproceedings{liu2021paint,
    title={Paint Transformer: Feed Forward Neural Painting with Stroke Prediction},
    author={Liu, Songhua and Lin, Tianwei and He, Dongliang and Li, Fu and Deng, Ruifeng and Li, Xin and Ding, Errui and Wang, Hao},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision},
    year={2021}
  }
  ```

## Acknowledgments

* This implementation is developed based on the code framework of **[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)** by Junyan Zhu *et al.*
