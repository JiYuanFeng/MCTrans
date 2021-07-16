<div align="center">
  <img src="imgs/logo.png" width="550"/>
</div>


## News
The code of MCTrans has been released. if you are interested in contributing to the standardization of the medical image analysis community, please feel free to contact me.

## Introduction 

- This repository provides code for "**Multi-Compound Transformer for Accurate Biomedical Image Segmentation**" [[paper](https://arxiv.org/pdf/2106.14385.pdf)].

- The MCTrans repository heavily references and uses the packages of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [MMCV](https://github.com/open-mmlab/mmcv), and [MONAI](https://monai.io/). We thank them for their selfless contributions

  

## Highlights

- A comprehensive toolbox for medical image segmentation, including flexible data loading, processing, modular network construction, and more.

- Supports representative and popular medical image segmentation methods, e.g. UNet, UNet++, CENet, AttentionUNet, etc. 

  

## Changelog
The first version was released on 2021.7.16.

## Model Zoo

Supported backbones:

- [x] VGG
- [x] ResNet

Supported methods:

- [x] UNet
- [x] UNet++
- [x] AttentionUNet
- [x] CENet
- [x] TransUNet
- [x] NonLocalUNet

## Installation and Usage

Please see the [guidance.md](docs/guidance.md).



## Citation

If you find this project useful in your research, please consider cite:

```latex
@article{ji2021multi,
  title={Multi-Compound Transformer for Accurate Biomedical Image Segmentation},
  author={Ji, Yuanfeng and Zhang, Ruimao and Wang, Huijie and Li, Zhen and Wu, Lingyun and Zhang, Shaoting and Luo, Ping},
  journal={arXiv preprint arXiv:2106.14385},
  year={2021}
}
```



## Contribution

I don't have a lot of time to improve the code base at this stage, so if you have some free time and are interested in contributing to the standardization of the medical image analysis community, please feel free to contact me (jyuanfeng8@gmail.com).



## License

This project is released under the [Apache 2.0 license](LICENSE).

