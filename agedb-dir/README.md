# RankSim on AgeDB-WIKI-DIR
This repository contains the implementation of __RankSim__ on *AgeDB-DIR* dataset. 

The imbalanced regression framework and LDS+FDS are based on the public repository of [Yang et al., ICML 2021](https://github.com/YyzHarry/imbalanced-regression/tree/main/agedb-dir). 

The blackbox combinatorial solver is based on the public repository of [Vlastelica et al., ICLR 2020](https://github.com/martius-lab/blackbox-backprop).

## Installation

#### Prerequisites

1. Download AgeDB dataset from [here](https://ibug.doc.ic.ac.uk/resources/agedb/) and extract the zip file (you may need to contact the authors of AgeDB dataset for the zip password) to folder `./data` 

2. We use the standard train/val/test split file (`agedb.csv` in folder `./data`) provided by Yang et al.(ICML 2021), which is used to set up balanced val/test set. To reproduce the results in the paper, please directly use this file. You can also generate it using

```bash
python data/create_agedb.py
python data/preprocess_agedb.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- tensorboard_logger
- numpy, pandas, scipy, tqdm, matplotlib, PIL, wget

## Code Overview

#### Main Files

- `train.py`: main training and evaluation script
- `create_agedb.py`: create AgeDB raw meta data
- `preprocess_agedb.py`: create AgeDB-DIR meta file `agedb.csv` with balanced val/test set

#### Main Arguments

- `--data_dir`: data directory to place data and meta file
- `--reweight`: cost-sensitive re-weighting scheme to use
- `--loss`: training loss type
- `--regularization_weight`: gamma, weight of the regularization term (default 100.0)
- `--interpolation_lambda`: lambda, interpolation strength parameter(default 2.0) 

## Getting Started

### 1. Train baselines

To use Vanilla model

```bash
python train.py --batch_size 256 --lr 1e-3
```
To use square-root inverse

```bash
python train.py  --batch_size 256 --lr 1e-3 --reweight sqrt_inv 
```

To use LDS (Yang et al., ICML 2021) with originally reported hyperparameters

```bash
python train.py  --batch_size 256 --lr 1e-3 --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2
```

To use FDS (Yang et al., ICML 2021) with originally reported hyperparameters

```bash
python train.py  --batch_size 256 --lr 1e-3 --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2
```

### 2. Train a model with RankSim
##### batch size 64, learning rate 2.5e-4
```bash
python train.py --batch_size 64 --lr 2.5e-4 --regularization_weight=100.0 --interpolation_lambda=2.0 
```


### 3. Train a model with RankSim and square-root frequency inverse

##### batch size 64, learning rate 2.5e-4
```bash
python train.py  --batch_size 64 --lr 2.5e-4 --reweight sqrt_inv --regularization_weight=100.0 --interpolation_lambda=2.0 
```


### 4. Train a model with RankSim and different loss (by default L1 loss)

To use RankSim with Focal-R loss

```bash
python train.py --loss focal_l1 --batch_size 64 --lr 2.5e-4 --regularization_weight=100.0 --interpolation_lambda=2.0 
```

### 5. Train a model with RankSim and LDS

To use RankSim (gamma: 100.0, lambda: 2.0) with Gaussian kernel (kernel size: 5, sigma: 2)

##### batch size 64, learning rate 2.5e-4
```bash
python train.py --batch_size 64 --lr 2.5e-4 --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --regularization_weight=100.0 --interpolation_lambda=2.0 
```

### 6. Train a model with RankSim and FDS

To use RankSim (gamma: 100.0, lambda: 2.0) with Gaussian kernel (kernel size: 5, sigma: 2)

##### batch size 64, learning rate 2.5e-4
```bash
python train.py --batch_size 64 --lr 2.5e-4 --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --regularization_weight=100.0 --interpolation_lambda=2.0 
```

### 7. Train a model with RankSim and LDS + FDS

To use RankSim (gamma: 100.0, lambda: 2.0) with LDS (Gaussian kernel, kernel size: 5, sigma: 2) and FDS (Gaussian kernel, kernel size: 5, sigma: 2)

##### batch size 64, learning rate 2.5e-4
```bash
python train.py --batch_size 64 --lr 2.5e-4 --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --regularization_weight=100.0 --interpolation_lambda=2.0 
```
#### NOTE: We report the results with batch size of 64 & learn rate of 2.5e-4. You can try the batch size reported by Yang et al., ICML 2021 by changing the arguments, e.g. run SQINV + RankSim with batch size 256, learning rate 1e-3
```bash
python train.py  --batch_size 256 --lr 1e-3 --reweight sqrt_inv --regularization_weight=100.0 --interpolation_lambda=2.0 
```


### 8. Evaluate and reproduce

If you do not train the model, you can evaluate the model and reproduce our results directly using the pretrained weights from the anonymous links below.

```bash
python train.py --evaluate [...evaluation model arguments...] --resume <path_to_evaluation_ckpt>
```

__SQINV + RankSim__, MAE All 6.91 (*best MAE All-shot*)
[(weights)](https://drive.google.com/file/d/1NLCGNDA5vToe-MdOqokVYtUcJhGudIZe/view?usp=sharing) <br>

__SQINV + FDS + RankSim__, MAE Few-shot 9.68 (*best MAE Few-shot*)
[(weights)](https://drive.google.com/file/d/1-3S2itLDkXhaLgJ9JvhqvTV-mNvLOMT6/view?usp=sharing) <br>

__Focal-R + LDS + FDS + RankSim__, MAE Many-shot 6.17
[(weights)](https://drive.google.com/file/d/1vH_0NZE66DIiTSnsF2PGehD2AUJZ72uB/view?usp=sharing) <br>

__Focal-R + FDS + RankSim__, GM Med-shot 4.84
[(weights)](https://drive.google.com/file/d/1XeU-60RRLwpkolZRG45zsaLDG2fBwlwS/view?usp=sharing) <br>

__RRT + LDS + RankSim__, MAE Med-shot 7.54
[(weights)](https://drive.google.com/file/d/10kbUoJ7mT9KTiD_3nSnYL3Lrr6lyDNY6/view?usp=sharing)  <br>



