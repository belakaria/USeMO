## Uncertainty-Aware Search Framework for Multi-Objective Bayesian Optimization


This repository contains the python implementation for USeMO from the AAAI 2020 paper "[Uncertainty-Aware Search Framework for Multi-Objective Bayesian Optimization](https://aiide.org/ojs/index.php/AAAI/article/view/6561)". 

The implementation handles automatically the batch version of the algorithm by setting the variable "batch_size" to a number higher than 1. 


## Requirements
This code is implemented in Python and requires the following dependencies:

* [`sobol_seq`](https://github.com/naught101/sobol_seq) – for generating Sobol sequences
* [`platypus`](https://platypus.readthedocs.io/en/latest/getting-started.html#installing-platypus) – for multi-objective evolutionary algorithms
* [`scikit-learn`](https://scikit-learn.org/stable/modules/gaussian_process.html) – specifically `sklearn.gaussian_process` for GP modeling
* [`pygmo`](https://esa.github.io/pygmo2/install.html) – for parallel optimization algorithms

You can install the required packages using:

```bash
pip install sobol_seq platypus-opt scikit-learn pygmo
```
---
## Running USeMO

```bash
python main.py <function_names> <d> <seed> <initial_number> <total_iterations> <acquisation_name>
```

Here's an example command you could run from bash:

```bash
python main.py branin,Currin 2 0 5 100 ei
```

Explanation of arguments:

1. `function_names`: names of the benchmark functions separated by a comma
2. `d`: number of input dimensions 
3. `seed`: random seed 
4. `initial_number`: number of initial of evaluations
5. `total_iterations`: number of BO iterations
6. `acquisation_name`: The choice of the acquisation function 

---
### Citation
If you use this code please cite our paper:

```bibtex

@inproceedings{belakaria2020uncertainty,
  title={Uncertainty-aware search framework for multi-objective Bayesian optimization},
  author={Belakaria, Syrine and Deshwal, Aryan and Jayakodi, Nitthilan Kannappan and Doppa, Janardhan Rao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={06},
  pages={10044--10052},
  year={2020}
}

````

