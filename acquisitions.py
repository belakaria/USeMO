#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 08:56:45 2020

@author: syrinebelakaria
"""

from scipy.stats import norm
import math

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