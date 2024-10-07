# PDOL-optimal-control

This repository contains the source code for the paper "**_An Operator Learning Approach to Nonsmooth Optimal Control of Nonlinear PDEs_**" by Yongcun Song, Xiaoming Yuan, Hangrui Yue, and Tianyou Zeng.
The paper can be found at [arxiv:2409.14417](https://arxiv.org/abs/2409.14417).

## Requirements

To run the code in this repository, you will need following software and packages:

- [Python](https://www.python.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyTorch](https://pytorch.org/)

## Files

The name of the files suggests their functionality. For example:

- [`burgers_train_cts_deeponet_hc.py`](https://github.com/tianyouzeng/PDOL-optimal-control/blob/main/burgers_train_cts_deeponet_hc.py) is the source code for training the DeepONet model that approximates the control-to-state operator for the optimal control of stationary Burgers equations.
- [`semilinparab_train_gradadj_fno3d_hc.py`](https://github.com/tianyouzeng/PDOL-optimal-control/blob/main/semilinparab_train_gradadj_fno3d_hc.py) is the source code for training the FNO model that approximates the adjoint operator of the Fréchet derivative of the control-to-state operator in the optimal control of semilinear parabolic equations.
- [`bilinparab_optimize_fno3d.py`](https://github.com/tianyouzeng/PDOL-optimal-control/blob/main/bilinparab_optimize_fno3d.py) is the source code for solving the bilinear optimal control of parabolic equations by the trained FNO surrogate models.

Besides the files in the root directory:

- The [`models`](https://github.com/tianyouzeng/PDOL-optimal-control/tree/main/models) and [`utils`](https://github.com/tianyouzeng/PDOL-optimal-control/tree/main/utils) folders contain the class definitions of DeepONet, MIONet and FNO. They also contains utility classes and functions for training and evaluation.
- The [`data`](https://github.com/tianyouzeng/PDOL-optimal-control/tree/main/data) folder contains the source code we used for generating training sets and testing sets.
- The [`trained_models`](https://github.com/tianyouzeng/PDOL-optimal-control/tree/main/trained_models) folder contains the models trained by the code in this repository.
- The [`trad_alg`](https://github.com/tianyouzeng/PDOL-optimal-control/tree/main/trad_alg) folder contains the implementaion of some traditional numerical algorithms that we compared with in the paper.
- The [`env`](https://github.com/tianyouzeng/PDOL-optimal-control/tree/main/env) folder contains an example [conda](https://docs.conda.io/en/latest) environment for running the code.

## Citation

```
@misc{song2024operatorlearningapproachnonsmooth,
      title={An Operator Learning Approach to Nonsmooth Optimal Control of Nonlinear PDEs}, 
      author={Yongcun Song and Xiaoming Yuan and Hangrui Yue and Tianyou Zeng},
      year={2024},
      eprint={2409.14417},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2409.14417}, 
}
```