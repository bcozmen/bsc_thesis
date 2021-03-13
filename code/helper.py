import glob
import scipy.io
import numpy as np
import requests
from fhn import fc

#return_index = -1 return averaged data
#return_index = index
def load_matrices(directory = "../Greifswald_fMRI_DTI", matrix_type = "GW"):

    if matrix_type == "GW":
        CmatFilenames = glob.glob(directory + "/NAP_*/SC/Normalized/DTI_CM.mat")
        DmatFilenames = glob.glob(directory + "/NAP_*/SC/Normalized/DTI_LEN.mat")
        FmatFilenames = glob.glob(directory + "/NAP_*/FC/FC.mat")
        TmatFilenames = glob.glob(directory + "/NAP_*/FC/TC.mat")

    else:
        CmatFilenames = glob.glob(directory + "/sub-*/SC/DTI_CM.mat")
        DmatFilenames = glob.glob(directory + "/sub-*/SC/DTI_LEN.mat")
        FmatFilenames = glob.glob(directory + "/sub-*/B_TC_Matrix/Corr_Matrix.mat")
        TmatFilenames = glob.glob(directory + "/sub-*/B_TC_Matrix/Time_Course_Matrix.mat")


    CmatFilenames.sort()
    DmatFilenames.sort()
    FmatFilenames.sort()
    TmatFilenames.sort()
    
    Cmats = []
    Dmats = []
    Fmats = []
    Bolds = []

    for cm in CmatFilenames:
        this_cm = scipy.io.loadmat(cm)['sc']
        this_cm *= (0.011/np.mean(this_cm))
        Cmats.append(this_cm)
        
        
    for dm in DmatFilenames:
        this_dm = scipy.io.loadmat(dm)['len']
        this_dm *= (80. / np.mean(this_dm))
        Dmats.append(this_dm)

    for fm,bold in zip(FmatFilenames,TmatFilenames):
        try:
            Fmats.append(scipy.io.loadmat(fm)['fc'])
            Bolds.append(scipy.io.loadmat(bold)['tc'])
            continue
        except:
            pass

        try:
            Fmats.append(scipy.io.loadmat(fm)['corr_mat'])
            Bolds.append(scipy.io.loadmat(bold)['tc'])
        except:
            Fmats.append(np.load(fm))
            Bolds.append(np.load(bold))

    

    return Cmats, Dmats, Fmats, Bolds


