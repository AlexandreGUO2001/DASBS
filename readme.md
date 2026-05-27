# Discrete Adjoint Schrödinger Bridge Sampler (DASBS)

This is the official implementation of:

> **Discrete Adjoint Schrödinger Bridge Sampler**
> Wei Guo, Yuchen Zhu, Xiaochen Du, Juno Nam, Yongxin Chen, Rafael Gómez-Bombarelli, Guan-Horng Liu, Molei Tao, Jaemoo Choi.
> *ICML 2026.*

DASBS extends adjoint matching to discrete state spaces via a unified Schrödinger bridge / stochastic optimal control framework for CTMCs. The repository contains the code used to produce the main results in the paper.

## Installation

```bash
pip install torch torchvision
pip install hydra-core omegaconf wandb tqdm timm matplotlib pot
```

We use Python 3.12, PyTorch 2.x, and PyTorch's fused `scaled_dot_product_attention` backend (enabled automatically when calling `F.scaled_dot_product_attention`).

## Running the experiments

```bash
# Ising beta_high = 0.28
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    beta=0.28 L=24 \
    model.name=ropesit model.hidden_size=32 model.n_blocks=6 \
    batch_size=64 num_repl=8 buffer_size=512 \
    num_stages=5 num_steps.controller=400 num_steps.corrector=200 \
    resample_freq=20 sampling_steps=200

# Ising beta_crit = 0.4407
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    beta=0.4407 L=24 \
    model.name=ropesit model.hidden_size=32 model.n_blocks=6 \
    batch_size=64 num_repl=8 buffer_size=512 \
    num_stages=5 num_steps.controller=400 num_steps.corrector=200 \
    optim.lr=5e-4 \
    resample_freq=20 sampling_steps=200 \
    loss.corrector=dm beta_init=inf

# Ising beta_low = 0.6
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    beta=0.6 L=24 \
    model.name=ropesit model.hidden_size=32 model.n_blocks=6 \
    batch_size=64 num_repl=8 buffer_size=512 \
    num_stages=5 num_steps.controller=500 num_steps.corrector=250 \
    optim.lr=5e-4 \
    resample_freq=20 sampling_steps=200 \
    loss.corrector=dm beta_init=inf \
    noise.gamma=0.25

# Potts beta_high = 0.9
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    dist=potts beta=0.9 vocab_size=4 L=16 \
    model.name=ropesit model.hidden_size=32 model.n_blocks=6 \
    batch_size=128 num_repl=16 buffer_size=512 \
    num_stages=5 num_steps.controller=500 num_steps.corrector=250 \
    optim.lr=1e-3 \
    resample_freq=20 sampling_steps=100

# Potts beta_crit = 1.0986
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    dist=potts beta=1.0986 vocab_size=4 L=16 \
    model.name=ropesit model.hidden_size=32 model.n_blocks=6 \
    batch_size=128 num_repl=16 buffer_size=4096 \
    num_stages=5 num_steps.controller=200 num_steps.corrector=100 \
    optim.lr=5e-4 \
    resample_freq=10 sampling_steps=100 \
    loss.corrector=dm beta_init=inf

# Potts beta_low = 1.3
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    dist=potts beta=1.3 vocab_size=4 L=16 \
    model.name=ropesit model.hidden_size=32 model.n_blocks=6 \
    batch_size=128 num_repl=16 buffer_size=4096 \
    num_stages=5 num_steps.controller=500 num_steps.corrector=250 \
    optim.lr=1e-3 \
    resample_freq=20 sampling_steps=100 \
    loss.corrector=dm beta_init=inf \
    noise.gamma=0.25
```

## Citation

If you find our work useful, please consider citing this work as follows:

```bibtex
@inproceedings{guo2026discrete,
  title     = {Discrete Adjoint Schr\"odinger Bridge Sampler},
  author    = {Guo, Wei and Zhu, Yuchen and Du, Xiaochen and Nam, Juno and Chen, Yongxin and G\'omez-Bombarelli, Rafael and Liu, Guan-Horng and Tao, Molei and Choi, Jaemoo},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=G9KydTWzZL}
}
```

## Acknowledgements

This repository is developed based on [MDNS](https://github.com/yuchen-zhu-zyc/MDNS) under the MIT License.
