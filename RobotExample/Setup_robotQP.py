from casadi import *
import numpy as np
import networkx as netx
from collections import defaultdict
from control.matlab import dlqr
from scipy.sparse import lil_array

def setup_QP(A, B, Nsub, N, xd, alpha, beta, graph, lbx=-inf, ubx=inf, lbu=-inf, ubu=inf,*, solver_opts:dict|None = None):
    
    nx = A.shape[1]
    nu = B.shape[1]

    # ============================================ VARIABLES ==============================================================
    x = {}
    u = {}
    x0 = {}
    v = defaultdict(dict)
    lbz = defaultdict(dict)
    ubz = defaultdict(dict)

    for i in range(Nsub):
        
        x[i] = SX.sym(f'x_{i+1}', nx, N+1) # STATE
        u[i] = SX.sym(f'u_{i+1}', nu, N) # INPUT
        
        for j in range(Nsub):
            if (i,j) in graph.edges():
                v[i][j] = SX.sym(f'v_{i+1}{j+1}', nx, N+1) # STATE COPIES
                lbz[i][j] = lbx*np.ones([nx,N+1])
                ubz[i][j] = ubx*np.ones([nx,N+1])
            else:
                v[i][j] = np.zeros([nx,N+1])
        v[i][i] = x[i]
        lbz[i][i] = lbx*np.ones([nx,N+1])
        ubz[i][i] = ubx*np.ones([nx,N+1])

        x0[i] = SX.sym(f'x0_{i+1}',nx,1)

    # ============================================ OBJECTIVE FUNCTION ==============================================================
    J = {}
    Q_form = beta*netx.laplacian_matrix(graph)
    Q_track = alpha*np.eye(Nsub)

    C = Q_form + Q_track

    Q = np.kron(C, np.eye(nx))
    R = np.eye(nu)

    _,P,_ = dlqr(np.kron(A, np.eye(Nsub)), np.kron(B, np.eye(Nsub)), Q, np.kron(R, np.eye(Nsub)))
    
    for i in range(Nsub):
        J[i] = 0
        for k in range(N+1):
            Zi = []
            Xd = np.array([])
            for j in range(Nsub):
                Zi = vertcat(Zi, v[i][j][:,k])

                if (i,j) in graph.edges() or j == i:
                    Xd = np.concatenate((Xd, xd[:,j]))
                else:
                    Xd = np.concatenate((Xd, np.zeros(nx)))
            if k<N:
                J[i] += 0.5*(Zi-Xd).T @ Q @ (Zi-Xd) + 0.5* u[i][:,k].T @ R @ u[i][:,k]
            else:
                J[i] += 0.5*(Zi-Xd).T @ P @ (Zi-Xd)
            
    # ============================================ CONSTRAINTS : LOCAL ==============================================================
    g = {}
    lbg = {}
    ubg = {}
    lb_u = {}
    ub_u = {}

    for i in range(Nsub):
        g[i] = []
        lbg[i] = []
        ubg[i] = []    

        # INITIAL STATE CONSTRAINT
        g[i] = vertcat(g[i], x[i][:,0] - x0[i])
        lbg[i] = vertcat(lbg[i], np.zeros(nx))
        ubg[i] = vertcat(ubg[i], np.zeros(nx))

        # SYSTEM DYNAMICS CONSTRAINT
        for k in range(N):
            x_k = x[i][:,k]
            x_kp1 = x[i][:,k+1]
            u_k = u[i][:,k]

            g[i] = vertcat(g[i], x_kp1 - A@x_k - B@u_k)
            lbg[i] = vertcat(lbg[i], np.zeros(nx))
            ubg[i] = vertcat(ubg[i], np.zeros(nx))
        
        # INPUT BOUNDS
        lb_u[i] = []
        ub_u[i] = [] 
        for k in range(N):
            lb_u[i] = vertcat(lb_u[i], lbu*np.ones(nu))
            ub_u[i] = vertcat(ub_u[i], ubu*np.ones(nu))
        
    # ============================================ DECISION VARIABLES ==============================================================

    Z   = {}
    lbZ = {}
    ubZ = {}
    v_indices = {}
    x_indices = {}
    u_indices = {}

    for i in range(Nsub):
        Zi = []
        lbzi = []
        ubzi = []
        Ui = []
        offset = 0

        v_indices[i] = {}
        x_indices[i] = np.array([])
        for j in sorted([i] + list(netx.neighbors(graph, i))):
            v_indices[i][j] = np.array([])
            for k in range(N+1):
                if i != j:
                    idx = np.array(range(offset, offset + nx))
                    v_indices[i][j] = np.hstack((v_indices[i][j], idx))
                    offset += nx
                else:
                    idx = np.array(range(offset, offset + nx))
                    x_indices[i] = np.hstack((x_indices[i], idx))
                    offset += nx

                Zi = vertcat(Zi,v[i][j][:,k])
                lbzi = vertcat(lbzi, lbz[i][j][:,k])
                ubzi = vertcat(ubzi, ubz[i][j][:,k])
        
        u_indices[i] = np.array([])
        for k in range(N):
            u_idx = np.array(range(offset, offset + nu))
            u_indices[i] = np.hstack((u_indices[i], u_idx))
            offset += nu
            Ui = vertcat(Ui, u[i][:,k])

        Z[i]   = vertcat(Zi,Ui)
        lbZ[i] = vertcat(lbzi, lb_u[i])
        ubZ[i] = vertcat(ubzi, ub_u[i])
                
    # ============================================ CONSTRAINTS : CONSENSUS  ==============================================================

    n_cons = int((sum(dict(graph.degree()).values()).full() * nx * (N + 1))[0][0])

    E = {}
    for i in range(Nsub):
        E[i] = lil_array((n_cons, Z[i].shape[0]))

    rows = 0
    Ci = np.eye(nx)
    Di = 0*np.eye(nu)
    
    for i in range(Nsub):
        for j in netx.neighbors(graph, i):
            idx_vij = v_indices[i][j]
            idx_xj  = x_indices[j]
            idx_uj  = u_indices[j]

            Ei_row = lil_array((nx*(N+1), Z[i].shape[0]))
            Ej_row = lil_array((nx*(N+1), Z[j].shape[0]))

            Ei_row[:, idx_vij] = np.kron(np.eye(nx), np.eye(N+1))
            Ej_row[:, idx_xj] = np.kron(-Ci, np.eye(N+1))
            Ej_row[:, idx_uj] = np.kron(-Di, np.ones([N+1,N]))
            
            E[i][rows:rows+nx*(N+1),:] = Ei_row
            E[j][rows:rows+nx*(N+1),:] = Ej_row

            rows += nx*(N+1)

    # ============================================ CREATE QP SOLVERS  ==============================================================

    dQP = {}
    cQP = {}

    # Distributed
    lam = SX.sym('lam', n_cons)
    L = {}
    for i in range(Nsub):
        E_cas = SX(E[i].toarray())
        L[i] = J[i] + lam.T@E_cas@Z[i]
        params = vertcat(x0[i], lam)
        dQP[i] = {'x' : Z[i],
                  'f' : L[i],
                  'g' : vertcat(g[i]),
                  'p' : params}  
        if solver_opts is not None:
            dQP[i]['solver'] = qpsol(f'dQP_solver_{i+1}', 'qpoases', dQP[i], solver_opts)
        else:
            dQP[i]['solver'] = qpsol(f'dQP_solver_{i+1}', 'qpoases', dQP[i])
        dQP[i]['E'] = E[i]
        dQP[i]['cost_fcn'] = J[i]
        dQP[i]['lbx'] = lbZ[i]
        dQP[i]['ubx'] = ubZ[i]
        dQP[i]['lbg'] = lbg[i]
        dQP[i]['ubg'] = ubg[i]
        dQP[i]['idx_vij'] = v_indices[i]
        dQP[i]['idx_xi'] = x_indices[i]
        dQP[i]['idx_ui'] = u_indices[i]
    
    dQP['horizon'] = N
    dQP['graph'] = graph
    
    # Centeralized
    J_c = 0
    Z_c = []
    G_c = []
    params_c = []
    lbZ_c = []
    ubZ_c = []
    lbG_c = []
    ubG_c = []

    g_cons = 0    
    for i in range(Nsub):
        J_c += J[i]
        Z_c = vertcat(Z_c, dQP[i]['x'])
        G_c = vertcat(G_c, dQP[i]['g'])
        params_c = vertcat(params_c, x0[i][:])
        lbZ_c = vertcat(lbZ_c, lbZ[i][:])
        ubZ_c = vertcat(ubZ_c, ubZ[i][:])
        lbG_c = vertcat(lbG_c, lbg[i][:])
        ubG_c = vertcat(ubG_c, ubg[i][:])
        g_cons += E[i].toarray()@Z[i] 
    
    G_c = vertcat(G_c, g_cons[:])
    lbG_c = vertcat(lbG_c, np.zeros(n_cons))
    ubG_c = vertcat(ubG_c, np.zeros(n_cons))
    cQP = {'x': Z_c, 'f': J_c, 'g':vertcat(G_c), 'p':vertcat(params_c)}
    cQP['solver'] = qpsol('cQP_solver','qpoases',cQP)
    cQP['lbx'] = lbZ_c
    cQP['ubx'] = ubZ_c
    cQP['lbg'] = lbG_c
    cQP['ubg'] = ubG_c
    cQP['N'] = N
    cQP['sys'] = types.SimpleNamespace()
    cQP['sys'].A = A
    cQP['sys'].B = B
    numZ = [Z[i].shape[0] for i in range(Nsub)]
    cQP['primal_idx'] = np.array(numZ)
    cQP['graph'] = graph
    cQP['x_idx'] = x_indices
    cQP['u_idx'] = u_indices
    cQP['v_idx'] = v_indices
    cQP['n_cons'] = n_cons

    return dQP, cQP

def solve_QP_centrally(cQP,x0):
    lbg = cQP['lbg']
    ubg = cQP['ubg']
    Nsub = x0.shape[-1]

    param = x0.T.reshape(-1,1)
    
    raw_sol = cQP['solver'](p=param, lbg=lbg, ubg=ubg, lbx=cQP['lbx'], ubx=cQP['ubx'])
    start_idx = 0
    cSol = {}
    for i in range(Nsub):
        stop_idx =start_idx + cQP['primal_idx'][i]
        cSol[f'Z_{i+1}'] = raw_sol['x'][start_idx:stop_idx].full()
        start_idx = stop_idx  

        for j in sorted([i] + list(netx.neighbors(cQP['graph'], i))):
            if j==i:
                cSol[f'x{i+1}_traj'] = cSol[f'Z_{i+1}'][list(cQP['x_idx'][i].astype(int))].reshape(-1,2).T
            else:
                cSol[f'v{i+1}{j+1}_traj'] = cSol[f'Z_{i+1}'][cQP['v_idx'][i][j].astype(int)].reshape(-1,2).T
        
        cSol[f'u{i+1}_traj'] = cSol[f'Z_{i+1}'][cQP['u_idx'][i].astype(int)].reshape(-1,2).T
    
    cSol['lam'] = raw_sol['lam_g'][-(cQP['n_cons']):].full()

    return cSol



# if __name__ == '__main__':
#     # PROBLEM SETUP 
#     Nrobot = 4
#     Ts = 1
#     Tsim = 100*Ts
#     Tpred = 5*Ts
#     Nsim = int(Tsim/Ts)
#     N = int(Tpred/Ts)

#     # GRAPH STRUCTURE
#     graph_structure = "PATH" # 'PATH', "CYCLE" or "WHEEL"
#     if graph_structure == "PATH":
#         graph = netx.path_graph(Nrobot)
#     elif graph_structure == "CYCLE":
#         graph = netx.cycle_graph(Nrobot)
#     elif graph_structure == "WHEEL":
#         graph = netx.wheel_graph(Nrobot)
#     else:
#         print("Further graphs not implemented")

#     # TARGET POINTS
#     theta = np.arange(Nrobot)*(2*np.pi/Nrobot)
#     xd = np.round(np.vstack([np.cos(theta), np.sin(theta)]),2)

#     # INITIAL STATES/QP PARAMETERS
#     x0 = np.round(np.vstack([10*np.sin(theta), 10*np.cos(theta)]))

#     # QP 
#     nx = x0.shape[0]
#     nu = 2

#     Ad = np.eye(nx)  # System Dynamics
#     Bd = Ts*np.eye(nu)

#     x_min = -10
#     x_max = 10
#     u_min = -3
#     u_max = 3
#     alpha = 10 # Tracking weight
#     beta = 1e-1 # Inter-agent distance weight
#     dOcp, cOcp = setup_QP(A=Ad, B=Bd, Nsub=Nrobot, xd=xd, N=N, lbx=x_min, ubx=x_max, lbu=u_min, ubu=u_max, alpha = alpha, beta=beta, graph=graph)
