## Installation

a. Create a conda virtual environment and install required packages.

```shell
conda create -n mctrans pip python=3.7
conda activate mctrans
git clone https://github.com/JiYuanFeng/MCTrans.git
cd MCTrans
python setup.py develop
```

a. Complie other CUDA operators such as [MultiScaleDeformableAttention](https://github.com/fundamentalvision/Deformable-DETR).

```shell
cd mctrans/models/ops/
bash make.sh
```

c. Create data folder under the MCTrans and link the actual dataset path ($DATA_ROOT).

```shell
mkdir data
ln -s $DATA_ROOT data
```



## Datasets Preparation

- It is recommended to you to convert your dataset (espeacial the label) to standard format. For example, The binary segmengtaion label shoule only contain `0,1` or `0,255`. 

- If your folder structure is different, you may need to change the corresponding paths in config files.

- We have upload some preprocessed datasets at [drive](), you can download and unpack them under the data folder.

  ```none
  MCTrans
  ├── mctrans
  ├── data
  │   ├── pannuke
  │   │   ├── Images
  │   │   ├── Lables
  │   ├── cvc-clinic
  │   │   ├── Images
  │   │   ├── Lables
  │   ├── cvc-colondb
  │   │   ├── Images
  │   │   ├── Lables
  │   ├── kvasir
  │   │   ├── Images
  │   │   ├── Lables
  ```

## Single GPU Training
```shell
bash tools/train.sh
```

## Multi GPU Training

```none
TO DO
```
