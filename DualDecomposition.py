from casadi import *
import numpy as np
from scipy.linalg import block_diag, svd, fractional_matrix_power

def DD_FGM(dOcp, cSol, p0, lam0, opts=None, *, init_guess=None, eta : float = 0.5):
    if opts is None:
        opts = dict()
        opts['max_iter'] = 5000
        opts['tol'] = 1e-9


    Nsub = p0.shape[1]
    ncons = dOcp[0]['E'].shape[0]
    init_params = {}

    if lam0.size > 0 :
        init_params['lam'] = lam0
    else:
        init_params['lam'] = np.zeros(ncons)

    try:
        init_params['p'] = p0
    except:
        print('Please provide initial parameters for QP')

    # STEP SIZE CALCULATION
    min_eig = np.inf
    E = None
    H_fun ={}
    H = {}
    for i in range(Nsub):
        xs = np.zeros(dOcp[i]['x'].shape)
        H_fun[i] = Function('Hess_fun',
                            [dOcp[i]['x']],
                            [hessian(dOcp[i]['cost_fcn'], dOcp[i]['x'])[0]])
        H[i] = H_fun[i](xs).full()
        min_eig = min(min_eig, np.min(np.linalg.eigvals(H[i])))
        if E is None:
            try:
                E = dOcp[i]['E'].toarray()
            except:
                E = dOcp[i]['E']
        else:
            try:
                E = horzcat(E, dOcp[i]['E'].toarray())
            except:
                E = horzcat(E, dOcp[i]['E'])
    
    H_blk = block_diag(*[H[i] for i in range(Nsub)])
    L = np.max(svd(E.full()@fractional_matrix_power(H_blk, -0.5), compute_uv=False))

    # DD LOOP
    opt_gap = dict()
    dSol = dict()
    dSol['iter'] = 0
    viol_arr = []
    
    mu = init_params['lam']
    lamb = init_params['lam']
    alpha  = 0.5*(np.sqrt(5)-1) # alpha_init = 1

    while True :
        # SOLVE PRIMAL SUBPROBLEMS
        for i in range(Nsub):
            if dSol['iter'] == 0:
                dSol[i] = dict()
                dSol[i]['opt_gap'] = np.array([])
            try:
                params_i = vertcat(*init_params['p'][:,i], lamb).full()
            except:
                print('Dimension issue of lambda')
                break

            raw_sol = dOcp[i]['solver'](p=params_i, lbg=dOcp[i]['lbg'], ubg=dOcp[i]['ubg'], lbx=dOcp[i]['lbx'], ubx=dOcp[i]['ubx'])
            dSol[i]['x'] = raw_sol['x'].full()

            if cSol:
                try:
                    opt_gap[i] = np.linalg.norm(dSol[i]['x']- cSol[f'theta_{i+1}'],inf)
                except:
                    opt_gap[i] = np.linalg.norm(dSol[i]['x']- cSol[f'Z_{i+1}'],inf)
                
                dSol[i]['opt_gap'] = np.append(dSol[i]['opt_gap'], opt_gap[i])
        
        # CALCULATE CONSENSUS VIOLATION
        res = 0
        for i in range(Nsub):
            res += dOcp[i]['E']@dSol[i]['x']

        dSol['iter'] += 1
        dSol['consensus_violation'] = np.linalg.norm(res, inf)
        viol_arr.append(dSol['consensus_violation'])

        # IF CONVERGED, BREAK
        if dSol['consensus_violation'] < opts['tol'] or dSol['iter'] >= opts['max_iter']:
            dSol['lam'] = lamb
            break 
        # ELSE, UPDATE DUAL VARIABLES
        else:
            lambda_kp1 = mu + (1/L)*res.flatten()
            alpha_kp1 = 0.5*alpha*(np.sqrt(alpha**2 +4)-alpha)
            beta = eta*alpha*(1-alpha)/(alpha**2+alpha_kp1)

            mu_kp1 = lambda_kp1 + beta*(lambda_kp1-lamb); #Dual Update
            mu = mu_kp1
            lamb = lambda_kp1
            alpha = alpha_kp1    
    
    dSol['consensus_array'] = viol_arr
    return dSol


def DD_AGM(dQP, cSol, p0, lam0, opts=None, *, init_guess=None, eta: float = 0.5):

    if opts is None:
        opts = dict()
        opts['max_iter'] = 5000
        opts['tol'] = 1e-9


    Nsub = p0.shape[1]
    ncons = dQP[0]['E'].shape[0]
    init_params = {}

    if lam0.size > 0 :
        init_params['lam'] = lam0
    else:
        init_params['lam'] = np.zeros(ncons)

    try:
        init_params['p'] = p0
    except:
        print('Please provide initial parameters for QP')

    # STEP SIZE CALCULATION
    min_eig = np.inf
    E = None
    H_fun ={}
    H = {}
    for i in range(Nsub):
        xs = np.zeros(dQP[i]['x'].shape)
        H_fun[i] = Function('Hess_fun',
                            [dQP[i]['x']],
                            [hessian(dQP[i]['cost_fcn'], dQP[i]['x'])[0]])
        H[i] = H_fun[i](xs).full()
        min_eig = min(min_eig, np.min(np.linalg.eigvals(H[i])))
        if E is None:
            try:
                E = dQP[i]['E'].toarray()
            except:
                E = dQP[i]['E']
        else:
            try:
                E = horzcat(E, dQP[i]['E'].toarray())
            except:
                E = horzcat(E, dQP[i]['E'])
    
    H_blk = block_diag(*[H[i] for i in range(Nsub)])
    s = np.max(svd(E.full()@fractional_matrix_power(H_blk, -0.5), compute_uv=False))
    L = s**2 
    
    # DD LOOP
    opt_gap = dict()
    dSol = dict()
    dSol['iter'] = 0
    viol_arr = []
    
    lam = init_params['lam']
    nu = init_params['lam']
    lam_km1 = lam.copy()

    while True :

        # ACCELERATE DUAL VARIABLE
        k = dSol['iter']
        beta = eta*(k-1)/(k+2) if k>=1 else 0.0
        nu = lam + beta*(lam - lam_km1)
        
        # SOLVE PRIMAL SUBPROBLEMS
        for i in range(Nsub):
            if dSol['iter'] == 0:
                dSol[i] = dict()
                dSol[i]['opt_gap'] = np.array([])
            try:
                params_i = vertcat(*init_params['p'][:,i], nu).full()
            except:
                print('Dimension issue of lambda')
                break

            raw_sol = dQP[i]['solver'](p=params_i, lbg=dQP[i]['lbg'], ubg=dQP[i]['ubg'], lbx=dQP[i]['lbx'], ubx=dQP[i]['ubx'])
            dSol[i]['x'] = raw_sol['x'].full()

            if cSol:
                try:
                    opt_gap[i] = np.linalg.norm(dSol[i]['x']- cSol[f'theta_{i+1}'],inf)
                except:
                    opt_gap[i] = np.linalg.norm(dSol[i]['x']- cSol[f'Z_{i+1}'],inf)
                
                dSol[i]['opt_gap'] = np.append(dSol[i]['opt_gap'], opt_gap[i])
        
        # CALCULATE CONSENSUS VIOLATION
        res = 0
        for i in range(Nsub):
            res += dQP[i]['E']@dSol[i]['x']

        dSol['iter'] += 1
        dSol['consensus_violation'] = np.linalg.norm(res, inf)
        viol_arr.append(dSol['consensus_violation'])

        # IF CONVERGED, BREAK
        if dSol['consensus_violation'] < opts['tol'] or dSol['iter'] >= opts['max_iter']:
            dSol['lam'] = lam
            break 
        # ELSE, UPDATE DUAL VARIABLES
        else:
            lam_km1 = lam
            lam = nu + (1/L)*res.flatten()

    
    dSol['consensus_array'] = viol_arr
    return dSol
    

def DD(dOcp, cSol, p0, lam0, opts=None, init_guess=None):
    
    if opts is None:
        opts = dict()
        opts['max_iter'] = 5000
        opts['tol'] = 1e-9


    Nsub = p0.shape[1]
    ncons = dOcp[0]['E'].shape[0]
    init_params = {}

    if lam0.size > 0 :
        init_params['lam'] = lam0
    else:
        init_params['lam'] = np.zeros(ncons)

    try:
        init_params['p'] = p0
    except:
        print('Please provide initial parameters for QP')

    # STEP SIZE CALCULATION
    alpha = np.inf 
    E = None

    H_fun ={}
    H = {} 

    for i in range(Nsub):
        # EVALUATE HESSIAN 
        xs = np.zeros(dOcp[i]['x'].shape)
        H_fun[i] = Function('Hess_fun',
                            [dOcp[i]['x']],
                            [hessian(dOcp[i]['f'], dOcp[i]['x'])[0]])
        H[i] = H_fun[i](xs).full()

        alpha = min(alpha, np.min(np.linalg.eigvals(H[i])))

        if E is None:
            try:
                E = dOcp[i]['E'].toarray()
            except:
                E = dOcp[i]['E']
        else:
            try:
                E = horzcat(E, dOcp[i]['E'].toarray())
            except:
                E = horzcat(E, dOcp[i]['E'])

    L = np.linalg.norm(E.full(), 2)**2 
    c_max = 2*alpha/L 
    c = 0.99*c_max
    viol_arr = []
    # DD LOOP 

    opt_gap = dict()
    dSol = dict()
    lam = init_params['lam']
    dSol['iter'] = 0

    while True :
        # SOLVE PRIMAL SUBPROBLEMS
        for i in range(Nsub):
            if dSol['iter'] == 0:
                dSol[i] = dict()
                dSol[i]['opt_gap'] = np.array([])

            params_i = vertcat(*init_params['p'][:,i], lam).full()
            raw_sol = dOcp[i]['solver'](p=params_i, lbg=dOcp[i]['lbg'], ubg=dOcp[i]['ubg'], lbx=dOcp[i]['lbx'], ubx=dOcp[i]['ubx'])
            dSol[i]['x'] = raw_sol['x'].full()

            if cSol:
                try:
                    opt_gap[i] = np.linalg.norm(dSol[i]['x']- cSol[f'theta_{i+1}'],inf)
                except:
                    opt_gap[i] = np.linalg.norm(dSol[i]['x']- cSol[f'Z_{i+1}'],inf)
        
        # CALCULATE CONSENSUS VIOLATION
        res = 0
        for i in range(Nsub):
            res += dOcp[i]['E']@dSol[i]['x']

        dSol['iter'] += 1
        dSol['consensus_violation'] = np.linalg.norm(res, inf)
        viol_arr.append(dSol['consensus_violation'])


        # IF CONVERGED, BREAK
        if dSol['consensus_violation'] < opts['tol'] or dSol['iter'] >= opts['max_iter']:
            dSol['lam'] = lam
            break 
        # ELSE, UPDATE DUAL VARIABLES
        else:
            lam = lam + c*res.flatten()
    
    dSol['consensus_array'] = viol_arr
    return dSol

