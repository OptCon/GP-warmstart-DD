import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from DualDecomposition import DD, DD_FGM, DD_AGM
from scipy.stats import qmc
import networkx as netx
import numpy as np
from casadi import *
from SensorExample.Setup_sensorQP import setup_QP

def Solve_centrally(QP, param, ntheta, Ns):
    raw_sol = QP['solver'](p= param, lbg=0, ubg=0)
    
    cSol = dict()
    for i in range(Ns):
        cSol[f'theta_{i+1}'] = raw_sol['x'].full()[i*ntheta:(i+1)*ntheta]
    cSol['lam'] = raw_sol['lam_g'].full()

    return cSol

def collect_datapoints(Nsamp):

    # PROBLEM SETUP
    M = np.array(([1 ,0.1],
                  [0.1, 2]))  
    Ns = 4
    graph_structure = "PATH" # OR "CYCLE" or "WHEEL"
    ntheta = M.shape[0]

    if graph_structure == "PATH":
        grph = netx.path_graph(Ns)
    elif graph_structure == "CYCLE":
        grph = netx.cycle_graph(Ns)
    elif graph_structure == "WHEEL":
        grph = netx.wheel_graph(Ns)
    else:
        print("Further graphs not implemented")

    # SETUP OCPs
    qpOpts = {"terminationTolerance" : 1e-12}
    dQP, cQP = setup_QP(M, Ns, grph, solver_opts=qpOpts)
    
    # PARAMETER(INPUT) DATA 
    L = -2
    U = 2
    
    sampler = qmc.LatinHypercube(d=ntheta*Ns)
    X0 = sampler.random(n=Nsamp)
    
    X_i = L + (U - L) * X0
    
    # QP SOLUTION LOOP
    p0 = {}
    dd_opts = {'max_iter' : 5000,
                'tol' : 1e-10}
    
    n_cons = cQP['g'].shape[0] 
    problem_iters = []

    Y_i = np.zeros([Nsamp,n_cons])
    for i in range(Nsamp):
        
        # SOLVE QP CENTRALLY
        p = X_i[i,:]
        cSol = Solve_centrally(cQP,p,ntheta=ntheta, Ns=Ns)

        # VERIFY IF DD CONVERGES IN ONE STEP
        lam0 = cSol['lam']
        p = p.reshape([Ns, ntheta])
        p0 = p.T
        dSol = DD_AGM(dQP, cSol, p0, lam0,  dd_opts)

        if dSol['iter'] > 1:
            problem_iters.append(i)
            Y_i[i,:] = np.zeros([n_cons,1])
            break
        else:
            Y_i[i,:] = cSol['lam'].T
        
        print(f'Completed iteration {i+1}/{Nsamp}\n')
    
    return X_i, Y_i, problem_iters


if __name__ == "__main__":
    
    X_i, Y_i, faults = collect_datapoints(Nsamp=10000)
    if not faults:
        np.save(f'{PROJECT_ROOT}\\GP Regression\\GPassets\\data\\X_i_sensor_AGM.npy', X_i)
        np.save(f'{PROJECT_ROOT}\\GP Regression\\GPassets\\data\\Y_i_sensor_AGM.npy', Y_i)
    else:
        print(f'There was an issue in iteration {faults}.')