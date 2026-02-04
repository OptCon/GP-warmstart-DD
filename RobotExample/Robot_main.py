import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from RobotExample.Setup_robotQP import setup_QP, solve_QP_centrally
from DualDecomposition import DD, DD_FGM, DD_AGM
import numpy as np
import networkx as netx
from matplotlib import pyplot as plt
from casadi import vertcat



def main():
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
    xd = 0.5*np.round(np.vstack([np.cos(theta), np.sin(theta)]),2)

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
    qpOpts = {"terminationTolerance" : 1e-12}

    dQP, cQP = setup_QP(A=Ad, B=Bd, Nsub=Nrobot, xd= xd, N=N, lbx=x_min, ubx=x_max, lbu=u_min, ubu=u_max, 
                        alpha = alpha, beta =beta, graph=graph, solver_opts=qpOpts)

    cSol = solve_QP_centrally(cQP=cQP, x0=x0)

    coldstart = True
    if coldstart:
        lam0 = np.zeros(cSol['lam'].shape[0]).flatten()
    else:
        lam0 = cSol['lam'].flatten()

    # DUAL DECOMPOSITION
    dd_opts = {'max_iter' : 5000,
                'tol' : 1e-9}
    

    dSol_AGM = DD_AGM(dQP, cSol, p0=x0, lam0=lam0, opts=dd_opts, eta=0.3)
    dSol_FGM = DD_FGM(dQP, cSol, p0=x0, lam0=lam0, opts=dd_opts, eta=0.35)
    dSol_DD = DD(dQP, cSol, p0=x0, lam0=lam0, opts=dd_opts)

    print(f"Dual Decomposition with AGM converged in {dSol_AGM['iter']} iterations.")
    print(f"Dual Decomposition with FGM converged in {dSol_FGM['iter']} iterations.")
    print(f"Dual Decomposition converged in {dSol_DD['iter']} iterations.")

    plt.plot(dSol_AGM['consensus_array'],label="DD_AGM")
    plt.plot(dSol_FGM['consensus_array'],label="DD_FGM")
    plt.plot(dSol_DD['consensus_array'],label="DD")
    plt.legend()
    plt.yscale('log')
    plt.show()
    
if __name__ == "__main__":
    main()