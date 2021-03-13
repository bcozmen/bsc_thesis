import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io 
import time
from multiprocessing import Pool
import multiprocessing as mp
import multiprocessing as mp
import sys
import itertools
import os


from evaluate import evaluate
import helper as he
from fhn import load_parameters, fc, matrix_correlation,kolmogorov,fcd


def idx_to_data(x):
    if x in range(0,10):
        return x, "GW"
    elif x in range(10,25):
        return x-10, "HC"
    elif x in range(25,36):
        return x-25, "SCZ"

"""
def write_time(now, old_time, message = ""):
    temp = now - old_time
    hours = temp//3600
    temp = temp - 3600*hours
    minutes = temp//60
    seconds = temp - 60*minutes
    start_time = time.time()
    print(message + ' | execution time: %d:%d:%d' %(hours,minutes,seconds))
"""


NSUBJECTS = 1
CORES = 45

if len(sys.argv) != 3:
    raise Exception("Arguments are not correct")


MATRIX = int(sys.argv[1])
HALF = int(sys.argv[2])

if (HALF < 0) or (HALF > 3):
    raise Exception("2nd Argument is not correct")

SCs, LMs , FCs , BOLDs = he.load_matrices(directory = "../brain_data/Greifswald_fMRI_DTI")
SCs_hc, LMs_hc , FCs_hc , BOLDs_hc = he.load_matrices(directory = "../brain_data/HC_data", matrix_type="")
SCs_scz, LMs_scz , FCs_scz , BOLDs_scz = he.load_matrices(directory = "../brain_data/SCZ_data", matrix_type="")

SCs = SCs+ SCs_hc + SCs_scz
LMs = LMs + LMs_hc + LMs_scz
FCs = FCs+ FCs_hc + FCs_scz
BOLDs = BOLDs+ BOLDs_hc + BOLDs_scz


SC = SCs[MATRIX]
LM = LMs[MATRIX]
empFCDs = [fcd(k) for k in BOLDs]
NSUBJECTS = len(FCs)

duration = 390000
tot_partition = 2
if idx_to_data(MATRIX)[1] == "GW":
	duration = 810000
	tot_partition = 4


try:
    os.mkdir("data")
except:
    pass

try:
    os.mkdir("data/"+str(MATRIX))
except:
    pass

params = load_parameters()
params['dif'] = False
params['N'] = SC[0].shape[0]
params['duration'] = duration
params['ou_noise'] = False
params['dt'] = 0.1

#configure ranges
I_range, I_inc       = [0.2 , 1.2] , 41
K_range, K_inc       = [0.2 ,  1.2], 41
c_range, c_inc       = [20 , 100]   , 5
s_range, s_inc       = [-1, -4] , 3


I_spc = np.linspace(I_range[0], I_range[1], I_inc)
K_spc = np.linspace(K_range[0], K_range[1], K_inc)
c_spc = np.linspace(c_range[0], c_range[1], c_inc)
s_spc = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]



inds = [[(i1,p1),(i2,p2),(i3,p3),(i4,p4)] for i1,p1 in enumerate(I_spc) for i2,p2 in enumerate(K_spc) for i3,p3 in enumerate(c_spc) for i4,p4 in enumerate(s_spc)]
inds = inds[(HALF*len(inds))//(tot_partition)   :   ((HALF+1)*(len(inds))//(tot_partition))]
print(len(inds))


corr_out = np.ones(shape=(NSUBJECTS, len(I_spc), len(K_spc), len(c_spc), len(s_spc) )) *-2
KS_out = np.ones(shape=(NSUBJECTS, len(I_spc), len(K_spc), len(c_spc), len(s_spc) )) *-2

if MATRIX==0 and HALF==0:
    table = np.array([I_spc, K_spc, c_spc, s_spc], dtype=object )
    np.save("lu_table.npy", table, allow_pickle=True)


params['CM'] = SC
params['len'] = LM

inputs = [(ind, params, FCs, empFCDs, MATRIX, np.random.randint(4294967295)) for ind in inds]
outputs = []



for i in range(0, len(inputs), CORES):
    
    max_i = min(i+CORES,len(inputs))
    
    pool =  Pool(processes=CORES)
    r = pool.starmap_async(evaluate, inputs[i:max_i])
    
    pool.close()
    pool.join()
    r.wait()
    fitnesses = r.get()
    outputs += fitnesses



for inp,out in zip(inputs,outputs):
    ind = inp[0]
    out_cor = out[0]
    out_ks = out[1]
    corr_out[:, ind[0][0], ind[1][0], ind[2][0], ind[3][0]] = out_cor
    KS_out[:, ind[0][0], ind[1][0], ind[2][0], ind[3][0]] = out_ks


np.save("corr_out_" +str(MATRIX)+"_"+str(HALF)+".npy", corr_out)
np.save("ks_out_" +str(MATRIX)+"_"+str(HALF)+".npy", KS_out)



