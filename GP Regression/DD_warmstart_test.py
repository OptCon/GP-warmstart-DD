import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


from GPassets.preprocessing import PreProcessor
from GPassets.model import GPModel
import numpy as np
import networkx as netx
from RobotExample.Setup_robotQP import setup_QP as robotQP
from SensorExample.Setup_sensorQP import setup_QP as sensorQP
from SensorExample.Setup_sensorQP import Solve_centrally
from RobotExample.Robot_main import solve_QP_centrally
from scipy.stats import qmc
from DualDecomposition import DD_AGM
import time
from GPassets.plotting import *

def solve_Robot_DD(gp : GPModel | None = None , coldstart = True, X_i : np.ndarray | None = None):

    # Setup ROBOT QP
    Nrobot = 4
    Ts = 1
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

    # QP 
    nx = 2
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
    dQP, cQP = robotQP(A=Ad, B=Bd, Nsub=Nrobot, xd= xd, N=N, lbx=x_min, ubx=x_max, lbu=u_min, ubu=u_max, alpha = alpha, beta =beta, graph=graph, solver_opts=qpOpts)

    dd_opts = {'max_iter' : 5000,
                'tol' : 1e-8}

    iters = []
    opt_error = []
    t_infer = []
    for i in range(X_i.shape[0]):
        x0 = X_i[i,:].reshape([nx, Nrobot])
        cSol = solve_QP_centrally(cQP, x0)

        if coldstart:
            lam0 = np.zeros(cSol['lam'].shape[0]).flatten()
        else:
            t0 = time.perf_counter()
            lam0 = gp.infer(x_test_np=X_i[i,:], is_scaled=False)
            lam0 = lam0.flatten()
            t_infer.append(time.perf_counter() - t0)

        opt_error.append(np.max(np.abs(lam0 - cSol['lam'].flatten())))
        
        dSol = DD_AGM(dQP, cSol, x0, lam0, dd_opts, eta=0.3)

        iters.append(dSol['iter'])
    
    return np.array(iters), np.array(opt_error), np.array(t_infer)

def solve_Sensor_DD(gp : GPModel | None = None , coldstart = True, X_i : np.ndarray | None = None):
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
    dQP, cQP = sensorQP(M, Ns, graph=grph, solver_opts=qpOpts)

    dd_opts = {'max_iter' : 5000,
                'tol' : 1e-9}

    iters = []
    opt_error = []
    t_infer = []
    for i in range(X_i.shape[0]):
        x0 = X_i[i,:]
        cSol = Solve_centrally(cQP, x0, ntheta, Ns) 

        if coldstart:
            lam0 = np.zeros(cSol['lam'].shape[0]).flatten()
        else:
            t0 = time.perf_counter()
            lam0 = gp.infer(x_test_np=X_i[i,:], is_scaled=False)
            lam0 = lam0.flatten()
            t_infer.append(time.perf_counter() - t0)

        opt_error.append(np.max(np.abs(lam0 - cSol['lam'].flatten())))
        p0 = x0.reshape([Ns, ntheta]).T
        dSol = DD_AGM(dQP, cSol, p0, lam0, dd_opts, eta=0.65)

        iters.append(dSol['iter'])
    
    return np.array(iters), np.array(opt_error), np.array(t_infer)

def main(kernel, X_i, EXAMPLE):
    
    # Initialize GP Model
    kernel_type = kernel # "Matern12" OR "ReLU"
    version = 3 # 1 OR 2 OR 3 OR 4(1- default initialization, 2,3 - special initializations)
    device = 'cuda'
    depth = 1

    gp_config = {'kernel' : kernel_type,
                 'version': version,
                 'depth' : depth,
                }

    prep = PreProcessor()
    assets_path = f'{PROJECT_ROOT}\\GP Regression\\GPassets\\prep_assets_{EXAMPLE}.pkl'
    prep.load(assets_path)

    # Create GPModel object with GP config
    gp = GPModel(gp_config, prep, device)

    # Start up gp model
    model_savepath = f'{PROJECT_ROOT}\\GP Regression\\GPassets\\models\\{EXAMPLE}\\Model_{kernel_type}_{version}.pth' 
    gp.load_model(model_savepath)

    if EXAMPLE.casefold() == 'Robot Example'.casefold():
        ws_iters, ws_err, inference_time = solve_Robot_DD(gp = gp, coldstart= False, X_i = X_i)
        cs_iters, cs_err, _ = solve_Robot_DD(gp = None, coldstart=  True, X_i = X_i)
    
    elif EXAMPLE.casefold() == 'Sensor Example'.casefold():
        ws_iters, ws_err, inference_time = solve_Sensor_DD(gp = gp, coldstart= False, X_i = X_i)
        cs_iters, cs_err, _ = solve_Sensor_DD(gp = None, coldstart=  True, X_i = X_i)
        
    return ws_iters, ws_err, inference_time, cs_iters, cs_err


if __name__ == '__main__':

    EXAMPLE = 'Sensor Example' # "Sensor Example" or "Robot Example"
   
    # Generate new unseen initial states
    L = -2
    U = 2

    nx = 2
    Ns = 4
    sampler = qmc.LatinHypercube(d=nx*Ns, seed=0)
    X0 = sampler.random(n=50)

    X_i = L + (U - L) * X0

    k1 = 'Matern12'
    k2 = "ReLU"
    Matern_iters_warm, _, _, _, _ = main(k1, X_i, EXAMPLE)
    ReLU_iters_warm, ReLU_errors_warm, t_inf, iters_cold, errors_cold = main(k2, X_i, EXAMPLE)
    
    f1 = f'{PROJECT_ROOT}\\GP Regression\\GPassets\\data\\Matern_iters_{EXAMPLE}.npy' 
    f2 = f'{PROJECT_ROOT}\\GP Regression\\GPassets\\data\\ReLU_iters_{EXAMPLE}.npy'
    f3 = f'{PROJECT_ROOT}\\GP Regression\\GPassets\\data\\Cold_iters_{EXAMPLE}.npy'

    np.save(f1, Matern_iters_warm)
    np.save(f2, ReLU_iters_warm)
    np.save(f3,iters_cold)

    Matern_iters_warm = np.load(f1)
    ReLU_iters_warm = np.load(f2)
    iters_cold = np.load(f3)
    
    fname = f'{PROJECT_ROOT}\\GP Regression\\Resutls\\plots\\{EXAMPLE}\\Warmstart.pdf'
    plot_arrays(ar1 = ReLU_iters_warm, ar2= Matern_iters_warm ,ar3= iters_cold, filename=fname)

    # ReLUimprovement = iters_cold - ReLU_iters_warm
    # avg_imp = np.mean(ReLUimprovement)
    # var_imp = np.var(ReLUimprovement)
    # rel_impr = ReLUimprovement / iters_cold * 100
    # avg_rel = np.mean(rel_impr)
    # med_rel = np.median(rel_impr)
    # qs = np.quantile(ReLUimprovement, [0.0, 0.5, 1.0])
    # report_path = f'{PROJECT_ROOT}\\GP Regression\\reports\\{EXAMPLE}\\Warmstart_report_{k2}.txt'
    # lines = []
    # lines.append(f'===========================WARMSTART RESULTS:{k2} Kernel========================================')
    # lines.append(f"Average improvement with {k2} kernel = {avg_imp}")
    # lines.append(f"Variance of improvement with {k2} kernel = {var_imp}")
    # lines.append(f"Median improvement with {k2} kernel = {qs[1]:.4g}")
    # lines.append(f'Minimum improvement : {qs[0]:.4g}')
    # lines.append(f'Maximum improvement : {qs[2]:.4g}')
    # # lines.append(f"Average relative improvement with {k} kernel = {avg_rel}")
    # # lines.append(f"Median relative improvement with {k} kernel = {med_rel}")
    # report = "\n".join(lines)
    
    # with open(report_path, 'w', encoding='utf-8') as f:
    #             f.write(report)
    #             f.write("\n\n")

    # print(f'Average improvement with {k2} kernel = {avg_imp}')
    # print(f"Variance of improvement with {k2} kernel = {var_imp}")
    # print(f'Median improvement with {k2} kernel = {qs[1]:.4g}')

    # print(f'Average relative improvement with {k} kernel = {avg_rel}')
    # print(f'Median relative improvement with {k} kernel = {med_rel}')

    # fname = f'{PROJECT_ROOT}\\GP Regression\\plots\\{EXAMPLE}\\WS_{k}.pdf' 
    # plot_iters(ar1 = iters_warm, ar2= iters_cold, annotate=True, filename=fname)

    # k2 = "Matern12"
    # Matern_iters_warm, Matern_errors_warm, Matern_t_inf, iters_cold, E_Matern_cold = main(k2, X_i, EXAMPLE)
    # improvement = iters_cold - Matern_iters_warm
    # avg_imp = np.mean(improvement)
    # med_imp = np.median(improvement)
    # rel_impr = improvement / iters_cold * 100
    # avg_rel = np.mean(rel_impr)
    # med_rel = np.median(rel_impr)

    # print(f'Average improvement with {k2} kernel = {avg_imp}')
    # print(f'Median improvement with {k2} kernel = {med_imp}')
    # print(f'Average relative improvement with {k2} kernel = {avg_rel}')
    # print(f'Median relative improvement with {k2} kernel = {med_rel}')

    # fname = f'{PROJECT_ROOT}\\GP Regression\\plots\\{EXAMPLE}\\WS_{k2}.pdf' 
    # plot_iters(ar1 = Matern_iters_warm, ar2= iters_cold, annotate=True, filename=fname)


