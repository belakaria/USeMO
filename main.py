# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:34:01 2018

@author: Syrine Belakaria
"""
import os
import numpy as np
import math
from model import GaussianProcess
from scipy.stats import norm
import scipy
from platypus import NSGAII, Problem, Real
import sobol_seq
from pygmo import hypervolume

######################Algorithm input##############################
paths='.'

from benchmarks import branin,Currin
functions=[branin,Currin]
d=2

#from benchmarks import AckleyD,Rosen,Sphere
#functions=[AckleyD,Rosen,Sphere]
#d=5

#from benchmarks import Powell,Perm,Dixon,ZAKHAROV,RASTRIGIN,SumSquares
#functions=[Powell,Perm,Dixon,ZAKHAROV,RASTRIGIN,SumSquares]
#d=6

seed=0
np.random.seed(seed)
intial_number=5
referencePoint = [1e5]*len(functions)
bound=[0,1]

Fun_bounds = [bound]*d
grid = sobol_seq.i4_sobol_generate(d,1000,np.random.randint(0,1000))
design_index = np.random.randint(0, grid.shape[0])


############################acquisation functions#############################
def UCB(x,beta,GP):
    mean,std=GP.getPrediction(x)
    return mean + beta*std
def LCB(x,beta,GP):
    mean,std=GP.getPrediction(x)
    return mean - beta*std
def TS(x,beta,GP):
    mean=GP.model.sample_y(x.reshape(1,-1))
    mean=mean[0]
    return mean
def ei(x,beta,GP):
    mean,std=GP.getPrediction(x)
    y_best=min(GP.yValues)
    xi=1e-3    
    z = (y_best-xi-mean)/std
    return -1*(std*(z*norm.cdf(z) + norm.pdf(z)))
def pi(x,beta,GP):
    mean,std=GP.getPrediction(x)
    y_best=max(GP.yValues)
    xi=1e-3    
    z = (y_best-xi-mean)/std
    return -1*norm.cdf(z)
############function that gives the UCB variable at each iteration
def compute_beta(iter_num,d):
    mu=0.5
    tau=2*math.log((pow(iter_num,(d/2)+2)*pow(math.pi,2))/(3*0.05))
    beta=math.sqrt(mu*tau)
#    beta =0.125* np.log(2*iter_num+1)
    return beta
############################Set aquisation function 
acquisation=TS
batch_size=1 #In case you need batch version, you can set the batch size here 
###################GP Initialisation##########################

GPs=[]
for i in range(len(functions)):
    GPs.append(GaussianProcess(d))

for k in range(intial_number):
    exist=True
    while exist:
        design_index = np.random.randint(0, grid.shape[0])
        x_rand=list(grid[design_index : (design_index + 1), :][0])
        if (any((x_rand == x).all() for x in GPs[0].xValues))==False:
            exist=False
    for i in range(len(functions)):
        GPs[i].addSample(np.asarray(x_rand),np.round(functions[i](x_rand,d),decimals=4))

for i in range(len(functions)):   
    GPs[i].fitModel()

#### write the initial points into file
input_output= open(os.path.join(paths,'input_output.txt'), "a")
for j in range(len(GPs[0].yValues)):    
    input_output.write(str(GPs[0].xValues[j])+'---'+str([GPs[i].yValues[j] for i in range(len(functions))]) +'\n' )
input_output.close()

##################### main loop ##########

for l in range(100):

    beta=compute_beta(l+1,d)
    cheap_pareto_set=[]

    def CMO(x):
        global beta
        x=np.asarray(x)
        return [acquisation(x,beta,GPs[i])[0] for i in range(len(GPs))]
    
    problem = Problem(d, len(functions))
    problem.types[:] = Real(bound[0], bound[1])
    problem.function = CMO
    algorithm = NSGAII(problem)
    algorithm.run(2500)
    cheap_pareto_set=[solution.variables for solution in algorithm.result]
    cheap_pareto_set_unique=[]
    for i in range(len(cheap_pareto_set)):
        if (any((cheap_pareto_set[i] == x).all() for x in GPs[0].xValues))==False:
            cheap_pareto_set_unique.append(cheap_pareto_set[i])

    UBs=[[GPs[i].getPrediction(np.asarray(np.asarray(x)))[0][0]+beta*GPs[i].getPrediction(np.asarray(np.asarray(x)))[1][0] for i in range(len(GPs))] for x in cheap_pareto_set_unique]
    LBs=[[GPs[i].getPrediction(np.asarray(np.asarray(x)))[0][0]-beta*GPs[i].getPrediction(np.asarray(np.asarray(x)))[1][0] for i in range(len(GPs))] for x in cheap_pareto_set_unique]
    uncertaities= [scipy.spatial.Rectangle(UBs[i], LBs[i]).volume() for i in range(len(cheap_pareto_set_unique))]
    
    batch_indecies=np.argsort(uncertaities)[::-1][:batch_size]
    batch=[cheap_pareto_set_unique[i] for i in batch_indecies]


#---------------Updating and fitting the GPs-----------------   
    for x_best in batch:
        for i in range(len(functions)):
            GPs[i].addSample(np.asarray(x_best),np.round(functions[i](x_best,d),decimals=4))
            GPs[i].fitModel()
            
    ############################ write Input output into file ##################
    input_output= open(os.path.join(paths,'input_output.txt'), "a")    
    input_output.write(str(GPs[0].xValues[-1])+'---'+str([GPs[i].yValues[-1] for i in range(len(functions))]) +'\n' )
    input_output.close()
    
    ############################ write hypervolume into file##################
                
#    current_hypervolume= open(os.path.join(paths,'hypervolumes.txt'), "a")    
#    simple_pareto_front_evaluations=list(zip(*[GPs[i].yValues for i in range(len(functions))]))
#    print("hypervolume ", hypervolume((np.asarray(simple_pareto_front_evaluations))).compute(referencePoint))
#    current_hypervolume.write('%f \n' % hypervolume((np.asarray(simple_pareto_front_evaluations))).compute(referencePoint))
#    current_hypervolume.close()            
            
