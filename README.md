# Looking through the past

This is a PyTorch implementation of the continual learning experiments with deep neural networks described in the
following article:
* Looking through the past: better knowledge retention for generative replay in continual learning (under review)
  <https://arxiv.org/abs/2309.10012>

Short version of this work is presented on Continual Learning worskshop at ICCV2023.

This repository is based on continual learning replay:
<https://github.com/GMvandeVen/continual-learning>

Experiments are performed in the *academic continual learning setting*, whereby
a classification-based problem is split up into multiple, non-overlapping *contexts*
(or *tasks*, as they are often called) that must be learned sequentially.

## Installation & requirements
The current version of the code has been tested with `Python 3.10.4` on a Fedora operating system
with the following versions of PyTorch and Torchvision:
* `pytorch 1.11.0`
* `torchvision 0.12.0`

Further Python-packages used are listed in `requirements.txt`.
Assuming Python and pip are set up, these packages can be installed using:
```bash
pip install -r requirements.txt
```

The code in this repository itself does not need to be installed, but a number of scripts should be made executable:
```bash
chmod +x exps.sh exps_cycles.sh
```

## Demos
```bash
./exps.sh
```
This runs experiments for BIR, BIR+SI and other methods for comparison for seed=1.
Run this command on develop branch.

```bash
./exps_cycles.sh
```
This runs experiments for our method for seed=1.
Run this command on laten_match_dist_cycle branch.

### Citation
Please consider citing our papers if you use this code in your research:
```
@article{khan2023looking,
  title={Looking through the past: better knowledge retention for generative replay in continual learning},
  author={Khan, Valeriya and Cygert, Sebastian and Deja, Kamil and Trzci{\'n}ski, Tomasz and Twardowski, Bart{\l}omiej},
  journal={arXiv preprint arXiv:2309.10012},
  year={2023}
}
```