import numpy as np
import matplotlib.pyplot as plt
import random
import time 

from fhn import chunkwise_int, matrix_correlation, fc, kolmogorov, fcd, kolmogorov_fcd


def evaluate(individual,params, FCs, empFCDs, SC_index, randstate):

    params['max_bold'] = 60.
    
    params['I'] = individual[0][1]
    params['K'] = individual[1][1]
    params['c'] = individual[2][1]
    params['sigma'] = individual[3][1]


    
    ix1 = individual[0][0]
    ix2 = individual[1][0]
    ix3 = individual[2][0]
    ix4 = individual[3][0]
    path = "data/" + str(SC_index) + "/" + str(SC_index) + "_" + str(ix1) + "_" + str(ix2) + "_" + str(ix3) + "_" + str(ix4) + ".npy"
    

    local_rnd = np.random.RandomState(randstate)


    BOLD = chunkwise_int(params, randomseed=local_rnd.randint(4294967295))
    
    np.save(path,BOLD)
    
    if BOLD is None:
        return np.ones(len(FCs))*-2, np.ones(len(FCs))*-2


    cors = np.zeros(len(FCs))
    kss = np.zeros(len(FCs))


    FC = fc(BOLD)
    FCD = fcd(BOLD)
        
    for idx, empMat in enumerate(FCs):
        cors[idx] = matrix_correlation(FC, empMat)

    for idx, empFCD in enumerate(empFCDs):
        kss[idx] = kolmogorov_fcd(FCD, empFCD)
    return cors, kss
    
