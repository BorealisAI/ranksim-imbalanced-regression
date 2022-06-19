# RankSim on IMDB-WIKI-DIR
This repository contains the implementation of __RankSim__ (ICML 2022) on *IMDB-WIKI-DIR* dataset. 

The imbalanced regression framework and LDS+FDS are based on the public repository of [Yang et al., ICML 2021](https://github.com/YyzHarry/imbalanced-regression/tree/main/imdb-wiki-dir). 

The blackbox combinatorial solver is based on the public repository of [Vlastelica et al., ICLR 2020](https://github.com/martius-lab/blackbox-backprop).

## Installation

#### Prerequisites

1. Download and extract IMDB faces and WIKI faces respectively using

```bash
python download_imdb_wiki.py
```

2. We use the standard train/val/test split file (`imdb_wiki.csv` in folder `./data`) provided by Yang et al.(ICML 2021), which is used to set up balanced val/test set. To reproduce the results in the paper, please directly use this file. You can also generate it using

```bash
python data/create_imdb_wiki.py
python data/preprocess_imdb_wiki.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- numpy, pandas, scipy, tqdm, matplotlib, PIL, wget

## Code Overview

#### Main Files

- `train.py`: main training and evaluation script
- `create_imdb_wiki.py`: create IMDB-WIKI raw meta data
- `preprocess_imdb_wiki.py`: create IMDB-WIKI-DIR meta file `imdb_wiki.csv` with balanced val/test set

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
To use square-root frequence inverse (SQINV)

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

```bash
python train.py --batch_size 256 --lr 1e-3 --regularization_weight=100.0 --interpolation_lambda=2.0 
```

### 3. Train a model with RankSim and square-root frequency inverse (SQINV)

```bash
python train.py  --batch_size 256 --lr 1e-3 --reweight sqrt_inv --regularization_weight=100.0 --interpolation_lambda=2.0 
```


### 4. Train a model with RankSim and different loss (by default $L1$ loss)

To use RankSim with Focal-R loss

```bash
python train.py --loss focal_l1 --batch_size 256 --lr 1e-3 --regularization_weight=100.0 --interpolation_lambda=2.0 
```

### 5. Train a model with RankSim and LDS

To use RankSim (gamma: 100.0, lambda: 2.0) with Gaussian kernel (kernel size: 5, sigma: 2)

```bash
python train.py --batch_size 256 --lr 1e-3 --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --regularization_weight=100.0 --interpolation_lambda=2.0 
```


### 6. Train a model with RankSim and FDS

To use RankSim (gamma: 100.0, lambda: 2.0) with Gaussian kernel (kernel size: 5, sigma: 2)


```bash
python train.py --batch_size 256 --lr 1e-3 --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --regularization_weight=100.0 --interpolation_lambda=2.0 
```


### 7. Train a model with RankSim and LDS + FDS

To use RankSim (gamma: 100.0, lambda: 2.0) with LDS (Gaussian kernel, kernel size: 5, sigma: 2) and FDS (Gaussian kernel, kernel size: 5, sigma: 2)

```bash
python train.py --batch_size 256 --lr 1e-3 --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --regularization_weight=100.0 --interpolation_lambda=2.0 
```

#### NOTE: We find different batch sizes (e.g. batch size of 64 & learn rate of 2.5e-4) sometimes can improve the performance. You can try different batch size by changing the arguments, e.g. run SQINV + RankSim with batch size 64, learning rate 2.5e-4

```bash
python train.py  --batch_size 64 --lr 2.5e-4 --reweight sqrt_inv --regularization_weight=100.0 --interpolation_lambda=2.0 
```

### 8. Evaluate and reproduce

If you do not train the model, you can evaluate the model and reproduce our results directly using the pretrained weights from the links below.

```bash
python train.py --evaluate [...evaluation model arguments...] --resume <path_to_evaluation_ckpt>
```

#### Pretrained weights 
__Focal-R + LDS + FDS + RankSim__, MAE All-shot 7.67
[(weights)](https://drive.google.com/file/d/1gEwrnsaO1A4I-e50NTW5fOnUSy3cnrx5/view?usp=sharing) <br>

__RRT + FDS + RankSim__, MAE All 7.35 (*best MAE All-shot*)
[(weights)](https://drive.google.com/file/d/1bKAP64Mx64HGasBzac4lSYcda6DuYIe7/view?usp=sharing) <br>

__SQINV + RankSim__, MAE All-shot 7.42
[(weights)](https://drive.google.com/file/d/1U1o2dXIXTLzyGAS10vtCtAW7DoXtT9FM/view?usp=sharing) <br>

__SQINV + LDS + FDS + RankSim__, MAE All 7.69, MAE Few-shot 21.43 (*best MAE Few-shot*)
[(weights)](https://drive.google.com/file/d/1R8v__UELcHO2zKP6dxF0aKxG2L266hy0/view?usp=sharing) <br>


