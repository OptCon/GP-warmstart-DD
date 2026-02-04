from casadi import *
import numpy as np
import networkx as nx

def setup_QP(M, Ns,graph, *, solver_opts: dict | None = None):
    
    theta = dict()
    y = dict()
    J = dict()
    ntheta = M.shape[0]
    for i in range(Ns):
        
        theta[i] = SX.sym(f'theta_{i+1}', ntheta) # VARIABLE
        y[i] = SX.sym(f'y_{i+1}', ntheta) # PARAMETER
        
    
        # OBJECTIVE FUNCTION
        J[i] = 0.5*sumsqr(M @ theta[i] - y[i])

    # CONSTRAINTS
    graph_edges = list(graph.edges())
    n_cons = int(len(graph_edges)*ntheta)
    E = {i: np.zeros([n_cons,theta[1].shape[0]]) for i in range(Ns)}
    
    k=0
    for (i,j) in graph.edges():
        E[i][k:k+theta[i].shape[0],:] = np.eye(theta[1].shape[0])
        E[j][k:k+theta[i].shape[0],:] = -np.eye(theta[1].shape[0])
        k += theta[i].shape[0]

    g_cons = 0
    for i in range(Ns):
        g_cons += E[i]@theta[i]
    G = g_cons

    # DISTRIBUTED OCP
    dOcp = dict()
    L = dict()
    lam = SX.sym('lam', n_cons)
    params = dict()
    for i in range(Ns):
        L[i] = J[i] + lam.T@(E[i]@theta[i])
        params[i] = vertcat(*[y[i]], lam)
        dOcp[i] = {'x' : theta[i],
                   'f' : L[i],
                   'g' : [],
                   'p' : params[i]}  

    # CENTRALIZED OCP
    cOCP = dict()
    Xc = vertcat(*[theta[i] for i in range(Ns)])
    Paramsc = vertcat(*[y[i] for i in range(Ns)])
    Jc = 0
    for i in range(Ns):
        Jc += J[i]
    
    cOCP = {'x' : Xc,
            'f' : Jc,
            'g' : G,
            'p' : Paramsc}  # casADi necessary keys
    
    # CREATE SOLVERS
    for i in range(Ns):
        if solver_opts is not None:
            dOcp[i]['solver'] = qpsol(f'dOCP_solver_{i+1}', 'qpoases', dOcp[i], solver_opts)
        else:
            dOcp[i]['solver'] = qpsol(f'dOCP_solver_{i+1}', 'qpoases', dOcp[i])
        dOcp[i]['cost_fcn'] = J[i]
        dOcp[i]['E'] = E[i]
        dOcp[i]['graph'] = graph
        dOcp[i]['lbg'] = []
        dOcp[i]['ubg'] = []
        dOcp[i]['lbx'] = -inf*np.ones(dOcp[i]['x'].shape)
        dOcp[i]['ubx'] = inf*np.ones(dOcp[i]['x'].shape)

    cOCP['solver'] = qpsol('cOCP_solver', 'qpoases', cOCP)

    return dOcp, cOCP

def Solve_centrally(QP, param, ntheta, Ns):
    raw_sol = QP['solver'](p= param, lbg=0, ubg=0)
    
    cSol = dict()
    for i in range(Ns):
        cSol[f'theta_{i+1}'] = raw_sol['x'].full()[i*ntheta:(i+1)*ntheta]
    cSol['lam'] = raw_sol['lam_g'].full()

    return cSol