import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from DualDecomposition import DD, DD_FGM, DD_AGM
from scipy.stats import qmc
import networkx as netx
import numpy as np
from casadi import *
from RobotExample.Setup_robotQP import setup_QP
from RobotExample.Robot_main import solve_QP_centrally

def collect_datapoints(Nsamp):

   # PROBLEM SETUP 
    Nrobot = 4
    Ts = 1
    Tsim = 100*Ts
    Tpred = 5*Ts
    N = int(Tpred/Ts)

    # GRAPH STRUCTURE
    graph_structure = "PATH" # "PATH" OR "CYCLE" or "WHEEL"
    if graph_structure == "PATH":
        graph = netx.path_graph(Nrobot)
    elif graph_structure == "CYCLE":
        graph = netx.cycle_graph(Nrobot)
    elif graph_structure == "WHEEL":
        graph = netx.wheel_graph(Nrobot)
    else:
        print("Further graphs not implemented")

    # TARGET POINTS
    theta = np.arange(Nrobot)*(2*np.pi/Nrobot)
    xd = np.round(np.vstack([np.cos(theta), np.sin(theta)]),2)

    # INITIAL STATES/QP PARAMETERS
    x0 = np.round(np.vstack([10*np.sin(theta), 10*np.cos(theta)]))

    # QP 
    nx = x0.shape[0]
    nu = 2

    Ad = np.eye(nx)  # System Dynamics
    Bd = Ts*np.eye(nu)

    x_min = -10
    x_max = 10
    u_min = -3
    u_max = 3
    alpha = 10 # Tracking weight
    beta = 1e-1 # Inter-agent distance weight

    dQP, cQP = setup_QP(A=Ad, B=Bd, Nsub=Nrobot, xd= xd, N=N, lbx=x_min, ubx=x_max, lbu=u_min, 
                        ubu=u_max, alpha = alpha, beta =beta, graph=graph)
    
    # PARAMETER(INPUT) DATA 
    L = -2
    U = 2
    
    sampler = qmc.LatinHypercube(d=nx*Nrobot)
    X0 = sampler.random(n=Nsamp)
    
    X_i = L + (U - L) * X0
    
    # QP SOLUTION LOOP
    p0 = {}
    dd_opts = {'max_iter' : 5000,
                'tol' : 1e-8}
    
    n_cons = cQP['n_cons'] 
    problem_iters = []

    Y_i = np.zeros([Nsamp,n_cons])
    for i in range(Nsamp):
        
        # SOLVE QP CENTRALLY
        p0 = X_i[i,:].reshape([nx, Nrobot])
        cSol = solve_QP_centrally(cQP=cQP, x0=p0)

        # VERIFY IF DD CONVERGES IN ONE STEP
        lam0 = cSol['lam'].flatten()

        try:
            dSol = DD_AGM(dQP, cSol, p0, lam0, dd_opts)
        except:
            print('Dual Decomposition algorithm failed')
            problem_iters.append(i)

        if dSol['iter'] > 1:
            problem_iters.append(i)
            print('dual decomposition did not converge in 1 step with optimal initialization')
            break
        else:
            Y_i[i,:] = cSol['lam'].T
        
        print(f'Completed iteration {i+1}/{Nsamp}\n')
    
    return X_i, Y_i, problem_iters


if __name__ == "__main__":
    
    X_i, Y_i, faults = collect_datapoints(Nsamp=10000)
    if not faults:
        np.save(f'{PROJECT_ROOT}\\GP Regression\\GPassets\\data\\X_i_robot_AGM.npy', X_i)
        np.save(f'{PROJECT_ROOT}\\GP Regression\\GPassets\\data\\Y_i_robot_AGM.npy', Y_i)
        print('no faults')
    else:
        print(f'There was an issue in iteration {faults}.')