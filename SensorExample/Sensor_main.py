import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from Setup_sensorQP import setup_QP
from DualDecomposition import DD, DD_FGM, DD_AGM
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

def main():
    # MEASUREMENT MATRIX
    M = np.array(([1 ,0.1, 0.1],
                  [0.1, 2, 0.1],
                  [0.1, 0.1, 1]))  
    Ns = 4
    g = nx.path_graph(Ns)
    qpOpts = {"terminationTolerance" : 1e-12}
    dOcp, cOcp = setup_QP(M, Ns,g, solver_opts=qpOpts)

    # RANDOM PARAMETERS  
    np.random.seed(0)
    p0 = dict()
    p0 = np.random.randn(dOcp[0]['x'].shape[0],Ns)

    # SOLVE CENTRALIZED PROBLEM
    ntheta = M.shape[0]
    raw_sol = cOcp['solver'](p= p0.reshape(-1, order='F'), lbg=0, ubg=0)
    cSol = dict()
    for i in range(Ns):
        cSol[f'theta_{i+1}'] = raw_sol['x'].full()[i*ntheta:(i+1)*ntheta]
    cSol['lam'] = raw_sol['lam_g'].full()

    # DUAL DECOMPOSITION
    dd_opts = {'max_iter' : 5000,
                'tol' : 1e-9}
    
    coldstart = True
    if coldstart:
        lam0 = np.zeros(cOcp['g'].shape[0])
    else:
        lam0 = raw_sol['lam_g'].full()

    dSol_AGM = DD_AGM(dOcp, cSol, p0, lam0 , opts=dd_opts, eta=0.65)
    dSol_FGM = DD_FGM(dOcp, cSol, p0, lam0 , opts=dd_opts, eta=0.5)
    dSol_DD = DD(dOcp, cSol, p0, lam0 , opts=dd_opts)

    print(f"Dual Decomposition with AGM converged in {dSol_AGM['iter']} iterations.")
    print(f"Dual Decomposition with FGM converged in {dSol_FGM['iter']} iterations.")
    print(f"Dual Decomposition converged in {dSol_DD['iter']} iterations.")

    plt.plot(dSol_AGM['consensus_array'],label="DD_AGM")
    plt.plot(dSol_FGM['consensus_array'],label="DD_FGM")
    plt.plot(dSol_DD['consensus_array'],label="DD")
    plt.legend()
    plt.yscale('log')
    plt.show()

    print(f'Central theta: {cSol["theta_1"].flatten()} \nDistributed theta: {dSol_AGM[0]["x"].flatten()}')

if __name__ == "__main__":
    main()