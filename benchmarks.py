import math
import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy

def Rosen(x1, d):
    x=list(4*np.asarray(x1)-2)
    sum_i = 0
    for i in range(d-1):
        sum_i =sum_i + (100 * ((x[i]**2) - x[i+1])**2 + (x[i] - 1)**2)
    return sum_i


def Sphere(x1,d):
    x=list(4*np.asarray(x1)-2)
    sum_i = 0
    for i in range(d):
        sum_i =sum_i + (x[i]**2)
    return  sum_i


def AckleyD(x1, d):
    x=list(4*np.asarray(x1)-2)
    sum_i = 0
    for i in range(d):
        sum_i = sum_i + x[i]*x[i]
    square_sum = sum_i/d
    sum_i = 0
    for i in range(d):
        sum_i = sum_i + math.cos(2*3.1416*x[i])
    cos_sum = sum_i/d
    f_original = -20.0*math.exp(-0.2*math.sqrt(square_sum)) - math.exp(cos_sum) + 20 + math.exp(1)
    return f_original
################################################

def Currin(x, d):
    return float(((1 - math.exp(-0.5*(1/x[1]))) * ((2300*pow(x[0],3) + 1900*x[0]*x[0] + 2092*x[0] + 60)/(100*pow(x[0],3) + 500*x[0]*x[0] + 4*x[0] + 20))))

def branin(x1,d):
    x=deepcopy(x1)
    x[0]= 15* x[0]-5
    x[1]=15*x[1]
    return float(np.square(x[1] - (5.1/(4*np.square(math.pi)))*np.square(x[0]) + (5/math.pi)*x[0]- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x[0]) + 10)
################################################
def Powell(xx,d):

    vmin=-4
    vmax=5
    x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
    f_original=0
    for i in range(1,int(math.floor(d/4)+1)):
        f_original=f_original+pow(x[4*i-3]+10*x[4*i-2],2)+5*pow(x[4*i-1]-x[4*i],2)+pow(x[4*i-2]-2*x[4*i-1],4)+10*pow(x[4*i-3]-2*x[4*i],4)
    return float(f_original)

def Perm(xx,d):
 
    vmin=-1*d
    vmax=d
    beta=10
    x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
    f_original=0
    for i in range(1,d+1):
        sum1=0
        for j in range(1,d+1):
            sum1=sum1+(j+beta)*(x[j]-math.pow(j,-1*i))        
        f_original=f_original+math.pow(sum1,2)
    return f_original

def Dixon(xx,d):
    vmin=-10
    vmax=10
    x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
    f_original=0
    for i in range(2,d+1):    
        f_original=f_original+i*math.pow(2*math.pow(x[i],2)-x[i-1],2)
    f_original=f_original+math.pow(x[1]-1,1)
    return f_original
def ZAKHAROV(xx,d):
    vmin=-5
    vmax=10
    x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
    term1=0
    term2=0
    for i in range(1,d+1):
        term1=term1+x[i]**2
        term2=term2+0.5*i*x[i]
    f_original=term1+math.pow(term2,2)+math.pow(term2,4)
    return f_original
def RASTRIGIN(xx,d):
    vmin=-5.12
    vmax=5.12
    x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
    f_original=0
    for i in range(1,d+1):
        f_original=f_original+(x[i]**2-10*math.cos(2*x[i]*math.pi))
    f_original=f_original+10*d
    return f_original
def SumSquares(xx,d):
    vmin=-5.12
    vmax=5.12
    x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
    f_original=0
    for i in range(1,d+1):
        f_original=f_original+(i*math.pow(x[i],2))
    return f_original
################################################
def DTLZ14f_1(x, d):
    g=0
    for i in range(d):
        g=g+pow(x[i]-0.5,2)-math.cos(20*math.pi*(x[i]-0.5))    
    g=100*(d+g)
    y1=(1+g)*0.5*x[0]*x[1]*x[2]
    return y1
def DTLZ14f_2(x, d):
    g=0
    for i in range(d):
        g=g+pow(x[i]-0.5,2)-math.cos(20*math.pi*(x[i]-0.5))    
    g=100*(d+g)
    y2=(1+g)*0.5*(1-x[2])*x[0]*x[1]
    return y2
def DTLZ14f_3(x, d):
    g=0
    for i in range(d):
        g=g+pow(x[i]-0.5,2)-math.cos(20*math.pi*(x[i]-0.5))    
    g=100*(d+g)
    y3=(1+g)*0.5*(1-x[1])*x[0]
    return y3
def DTLZ14f_4(x, d):
    g=0
    for i in range(d):
        g=g+pow(x[i]-0.5,2)-math.cos(20*math.pi*(x[i]-0.5))    
    g=100*(d+g)
    y4=(1+g)*0.5*(1-x[0])
    return y4

#########################################
#d=4
def ZDT1_1(x, d):
    y1=x[0]
    return y1
def ZDT1_2(x, d):
    y1=x[0]
    g=0
    for i in range(1,d):
        g=g+x[i] 
    g=g*(9./(d-1))+1
    h=1-math.sqrt(y1)/math.sqrt(g)
    y2=g*h
    return y2  
###########################################