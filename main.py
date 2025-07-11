# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:34:01 2018

@author: Syrine Belakaria
"""
import os
import sys

import numpy as np
import scipy
import sobol_seq
from benchmark_functions import get_benchmark_functions
from platypus import NSGAII, Problem, Real
from pygmo import hypervolume

from acquisitions import compute_beta, get_acquisation_function
from model import GaussianProcess

######################Algorithm input##############################

args = sys.argv[1:]
function_names = args[0]
functions = get_benchmark_functions(function_names)
d = int(args[1])
seed = int(args[2])
intial_number = int(args[3])
total_iterations = int(args[4])
acquisation_name = args[5]  # 'ei', 'UCB', 'LCB', 'TS', 'pi'
paths = "."
np.random.seed(seed)

###################Sobol Sequence Initialisation and search bounds##########################
bound = [0, 1]
Fun_bounds = [bound] * d
grid = sobol_seq.i4_sobol_generate(d, 1000, np.random.randint(0, 1000))
design_index = np.random.randint(0, grid.shape[0])


############################Set aquisation function
acquisation = get_acquisation_function(acquisation_name)
batch_size = 1  # In case you need batch version, you can set the batch size here
###################GP Initialisation##########################

GPs = []
for i in range(len(functions)):
    GPs.append(GaussianProcess(d))

for k in range(intial_number):
    exist = True
    while exist:
        design_index = np.random.randint(0, grid.shape[0])
        x_rand = list(grid[design_index : (design_index + 1), :][0])
        if (any((x_rand == x).all() for x in GPs[0].xValues)) == False:
            exist = False
    for i in range(len(functions)):
        GPs[i].addSample(
            np.asarray(x_rand), np.round(functions[i](x_rand, d), decimals=4)
        )

for i in range(len(functions)):
    GPs[i].fitModel()

#### write the initial points into file
input_output = open(os.path.join(paths, "input_output.txt"), "a")
for j in range(len(GPs[0].yValues)):
    input_output.write(
        str(GPs[0].xValues[j])
        + "---"
        + str([GPs[i].yValues[j] for i in range(len(functions))])
        + "\n"
    )
input_output.close()

##################### main loop ##########

for l in range(100):

    beta = compute_beta(l + 1, d)
    cheap_pareto_set = []

    def CMO(x):
        global beta
        x = np.asarray(x)
        return [acquisation(x, beta, GPs[i])[0] for i in range(len(GPs))]

    problem = Problem(d, len(functions))
    problem.types[:] = Real(bound[0], bound[1])
    problem.function = CMO
    algorithm = NSGAII(problem)
    algorithm.run(2500)
    cheap_pareto_set = [solution.variables for solution in algorithm.result]
    cheap_pareto_set_unique = []
    for i in range(len(cheap_pareto_set)):
        if (any((cheap_pareto_set[i] == x).all() for x in GPs[0].xValues)) == False:
            cheap_pareto_set_unique.append(cheap_pareto_set[i])

    UBs = [
        [
            GPs[i].getPrediction(np.asarray(np.asarray(x)))[0][0]
            + beta * GPs[i].getPrediction(np.asarray(np.asarray(x)))[1][0]
            for i in range(len(GPs))
        ]
        for x in cheap_pareto_set_unique
    ]
    LBs = [
        [
            GPs[i].getPrediction(np.asarray(np.asarray(x)))[0][0]
            - beta * GPs[i].getPrediction(np.asarray(np.asarray(x)))[1][0]
            for i in range(len(GPs))
        ]
        for x in cheap_pareto_set_unique
    ]
    uncertaities = [
        scipy.spatial.Rectangle(UBs[i], LBs[i]).volume()
        for i in range(len(cheap_pareto_set_unique))
    ]

    batch_indecies = np.argsort(uncertaities)[::-1][:batch_size]
    batch = [cheap_pareto_set_unique[i] for i in batch_indecies]

    # ---------------Updating and fitting the GPs-----------------
    for x_best in batch:
        for i in range(len(functions)):
            GPs[i].addSample(
                np.asarray(x_best), np.round(functions[i](x_best, d), decimals=4)
            )
            GPs[i].fitModel()

    ############################ write Input output into file ##################
    input_output = open(os.path.join(paths, "input_output.txt"), "a")
    input_output.write(
        str(GPs[0].xValues[-1])
        + "---"
        + str([GPs[i].yValues[-1] for i in range(len(functions))])
        + "\n"
    )
    input_output.close()

    ############################ write hypervolume into file##################
    referencePoint = [1e5] * len(functions)
    current_hypervolume = open(os.path.join(paths, "hypervolumes.txt"), "a")
    simple_pareto_front_evaluations = list(
        zip(*[GPs[i].yValues for i in range(len(functions))])
    )
    print(
        "hypervolume ",
        hypervolume((np.asarray(simple_pareto_front_evaluations))).compute(
            referencePoint
        ),
    )
    current_hypervolume.write(
        "%f \n"
        % hypervolume((np.asarray(simple_pareto_front_evaluations))).compute(
            referencePoint
        )
    )
    current_hypervolume.close()
