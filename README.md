## Uncertainty-Aware Search Framework for Multi-Objective Bayesian Optimization


This repository contains the python implementation for USeMO from the AAAI 2020 paper "[Uncertainty-Aware Search Framework for Multi-Objective Bayesian Optimization](https://aiide.org/ojs/index.php/AAAI/article/view/6561)". 

The implementation handles automatically the batch version of the algorithm by setting the variable "batch_size" to a number higher than 1. 


### Requirements
The code is implemented in Python and requires the following packages:
1. [sobol_seq](https://github.com/naught101/sobol_seq)

2. [platypus](https://platypus.readthedocs.io/en/latest/getting-started.html#installing-platypus)

3. [sklearn.gaussian_process](https://scikit-learn.org/stable/modules/gaussian_process.html)

4. [pygmo](https://esa.github.io/pygmo2/install.html) 

### Citation
If you use this library in your academic work please cite our paper:

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

