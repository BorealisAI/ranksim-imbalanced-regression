# RankSim on STS-B-DIR
This repository contains the implementation of *RankSim* (ICML 2022) on *STS-B-DIR* dataset. 

The imbalanced regression framework and LDS+FDS are based on the public repository of [Yang et al., ICML 2021](https://github.com/YyzHarry/imbalanced-regression/tree/main/imdb-wiki-dir). 

The blackbox combinatorial solver is based on the public repository of [Vlastelica et al., ICLR 2020](https://github.com/martius-lab/blackbox-backprop).

## Installation

#### Prerequisites

1. Download GloVe word embeddings (840B tokens, 300D vectors) using

```bash
python glove/download_glove.py
```

2. We use the standard file (`./glue_data/STS-B`) provided by Yang et al.(ICML 2021), which is used to set up balanced STS-B-DIR dataset. To reproduce the results in the paper, please directly use this file. If you want to try different balanced splits, you can delete the folder `./glue_data/STS-B` and run

```bash
python glue_data/create_sts.py
```


#### Dependencies

The required dependencies for this task are quite different to other three tasks, so it's better to create a new environment for this task. If you use conda, you can create the environment and install dependencies using the following commands:

```bash
conda create -n sts python=3.6
conda activate sts
# PyTorch 0.4 (required) + Cuda 9.2
conda install pytorch=0.4.1 cuda92 -c pytorch
# other dependencies
pip install -r requirements.txt
# The current latest "overrides" dependency installed along with allennlp 0.5.0 will now raise error. 
# We need to downgrade "overrides" version to 3.1.0
pip install overrides==3.1.0
```

## Code Overview

#### Main Files

- `train.py`: main training and evaluation script
- `create_sts.py`: download original STS-B dataset and create STS-B-DIR dataset with balanced val/test set 


#### Main Arguments

- `--data_dir`: data directory to place data and meta file
- `--val_interval`: number of iterations between validation checks
- `--patience`: patience (number of validation checks) for early stopping
- `--reweight`: cost-sensitive re-weighting scheme to use
- `--loss`: training loss type
- `--regularization_weight`: gamma, weight of the regularization term (default 3e-4)
- `--interpolation_lambda`: lambda, interpolation strength parameter(default 2.0) 

## Getting Started

### 1. Train baselines

To use Vanilla model

```bash
python train.py --reweight none
```
To use frequency inverse (INV)

```bash
python train.py --reweight inverse
```

To use LDS (Yang et al., ICML 2021) with originally reported hyperparameters

```bash
python train.py  --reweight inverse --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2
```

To use FDS (Yang et al., ICML 2021) with originally reported hyperparameters

```bash
python train.py --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2
```

### 2. Train a model with RankSim

```bash
python train.py --regularization_weight=3e-4 --interpolation_lambda=2 
```

### 3. Train a model with RankSim and frequency inverse (INV)

```bash
python train.py --regularization_weight=3e-4 --interpolation_lambda=2 --reweight inverse
```

### 4. Train a model with RankSim and different loss (by default $L1$ loss)

To use RankSim with Focal-R (MSE) loss 

```bash
python train.py --loss focal_mse --regularization_weight=3e-4 --interpolation_lambda=2 --reweight inverse
```

### 5. Train a model with RankSim and LDS

To use RankSim with Gaussian kernel 

```bash
python train.py --regularization_weight=3e-4 --interpolation_lambda=2 --reweight inverse --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 
```

### 6. Train a model with RankSim and FDS

To use RankSim with Gaussian kernel 

```bash
python train.py --regularization_weight=3e-4 --interpolation_lambda=2 --reweight inverse --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 
```

### 7. Train a model with RankSim and LDS + FDS

To use RankSim  with LDS  and FDS 

```bash
python train.py --regularization_weight=3e-4 --interpolation_lambda=2 --reweight inverse --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 
```

### 8. Evaluate and reproduce

If you do not train the model, you can evaluate the model and reproduce our results directly using the pretrained weights from the links below.

```bash
python train.py [...evaluation model arguments...] --evaluate --eval_model <path_to_evaluation_ckpt>
```

### 9. Pretrained weights 
__Vanilla + RankSim__, MSE Medium-shot 0.767, Pearson correlation 72.9 (*best Medium-shot*)
[(weights)](https://drive.google.com/file/d/1YfrRxSIlgQPFDtGodhcUfIjp-laSqm20/view?usp=sharing) <br>

__RRT + RankSim__, MSE All 0.865, Pearson correlation All 77.1 (*best All-shot*); MSE Few-shot 0.670, Pearson correlation Few-shot 86.1 (*best Few-shot*)
[(weights)](https://drive.google.com/file/d/1x-M8VhwCVvFvu36f_VwGn_UDzZOEPUjF/view?usp=sharing) <br>

__FDS + RankSim__, MSE Medium-shot 0.767
[(weights)](https://drive.google.com/file/d/1vjcIs2gXhQUidnLxgsCL0RoA59bklCTN/view?usp=sharing) <br>

__INV + LDS + RankSim__, MSE All 0.889, Pearson correlation All 76.2; MSE Medium-shot 0.849, Pearson correlation Few-shot 70.0; MSE All 0.690, Pearson correlation All 85.6
[(weights)](https://drive.google.com/file/d/12kVI2Mm5hcVXeBPikN9dBJkpPPrfHLmS/view?usp=sharing) <br>

