import numpy as np
import numba
import matplotlib.pyplot as plt
import scipy.stats
def chunkwise_int(params,randomseed=0, chunkSize=100000, xinit=np.zeros(94), yinit=np.zeros(94), X_BOLD=None, F_BOLD=None, Q_BOLD=None, V_BOLD=None):
    '''
    Run the interareal network simulation with the parametrization params, compute the corresponding BOLD signal
    and store the result ( currently only BOLD signal ) in the hdf5 file fname_out
    The simulation runs in chunks to save working memory.
    chunkSize corresponds to the size in ms of a single chunk
    '''

    # load parameters parameters from dictionary

    N = params['N']
    SC = params['CM']
    DM = params['len']
    dt = params['dt']
    duration = params['duration']

    #load FHN parameters
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    delta = params['delta']
    epsilon = params['epsilon']
    tau = params['tau']

    #params to optimize
    I = params['I']
    K = params['K']
    sigma = params['sigma']
    c = params['c']
    
    
    max_bold = params['max_bold']
    dif = params['dif']


    #BOLD parameters (Trivial)
    samplingRate_NDt = int(round(2000/dt))
    BOLD         = np.zeros((N, int((duration/dt)//(samplingRate_NDt)) ))

    if X_BOLD is None:
        X_BOLD       = np.ones((N,))   # Vasso dilatory signal

    if F_BOLD is None:
        F_BOLD       = np.ones((N,))   # Blood flow

    if Q_BOLD is None:
        Q_BOLD       = np.ones((N,))   # Deoxyhemoglobin

    if V_BOLD is None:
        V_BOLD       = np.ones((N,))   # Blood volume
     

    #Ballon-Windkessel model parameters (Deco 2013, Friston 2003):
    rho_b     = 0.34                  # Capillary resting net oxygen extraction (dimensionless)
    alpha_b   = 0.32                  # Grubb's vessel stiffness exponent (dimensionless)
    V0_b      = 0.02                  # Resting blood volume fraction (dimensionless)
    k1_b      = 7 * rho_b               # (dimensionless)
    k2_b      = 2.0                   # (dimensionless)
    k3_b      = 2 * rho_b - 0.2         # (dimensionless)
    Gamma_b   = 0.41 * np.ones((N,))  # Rate constant for autoregulatory feedback by blood flow (1/s)
    K_b       = 0.65 * np.ones((N,))  # Vasodilatory signal decay (1/s)
    Tau_b     = 0.98 * np.ones((N,))  # Transit time  (s)


    #compute delay matrix
    
    DM = computeDelayMatrixdt(DM,c,dt)        

    maxDelay = int(np.amax(DM))
    if maxDelay == 0:
        return None

    if (maxDelay)>(duration/dt):
        raise Exception('Simulation time is not long enough for such a low transmission speed, choose higher c or much higher duration')

    # return arrays
    idxLastT = 0    # Index of the last computed t
    lents = min( chunkSize + maxDelay, (duration - dt * idxLastT+ (maxDelay)* dt)/dt )
    lents = int(np.ceil(lents))
    xs = np.zeros((N, lents))
    ys = np.zeros((N, lents))



    if randomseed > 0:
        np.random.seed(randomseed)

    #configure initial state of the network
    #this is necessary since we are using a delay matrix

    local_rnd = np.random.RandomState(randomseed)    
    noise = local_rnd.standard_normal(size=(N, int(15000+maxDelay))) / np.sqrt(dt)
    
    if np.isnan(noise).any():
        return None


    xs_try, ys_try, min_xs, max_xs = get_min_max(dt, 15000+maxDelay, 
                              N, SC, DM, maxDelay,
                              alpha, beta, gamma, delta, epsilon, tau,
                              I, K, sigma, 
                              noise, # c is already included in DM
                              np.zeros((N,15000+maxDelay)), np.zeros((N,15000+maxDelay)), False)
    

    if np.isnan(xs_try).any():
        return None
    if max_xs-min_xs == 0:
        return None
    if 0 in xs_try.shape:
        return None
    """
    else:
        dt_interval = (max_xs-min_xs)*interval_coef
        max_xs += dt_interval
        min_xs -= dt_interval
    """
    
    """
    if xinit.all() == 0 and randomseed>0:
        xinit = np.random.rand(N)
        yinit = np.random.rand(N)
    """

    if xs_try.shape[1]>maxDelay and maxDelay > 0:
        xs[:,:maxDelay] = xs_try[:,-maxDelay:]
        ys[:,:maxDelay] = ys_try[:,-maxDelay:]
    elif maxDelay > 0 and xinit.shape == (N,):
        xs[:,:maxDelay] = np.tile(xinit,(maxDelay,1)).T
        ys[:,:maxDelay] = np.tile(yinit,(maxDelay,1)).T
    elif maxDelay>0:
        xs[:,:maxDelay] = xinit
        ys[:,:maxDelay] = yinit
    else:
        xs[:,0] = xinit
        ys[:,0] = yinit

    
    while dt * idxLastT< duration:

        currentChunkSize = min( chunkSize + maxDelay, (duration - dt * idxLastT+ (maxDelay)* dt)/dt ) # max_global_delay + 1

        
        noise = local_rnd.standard_normal(size=(N, int(currentChunkSize))) / np.sqrt(dt)

        if np.isnan(noise).any():
            return None

        xs,ys, BOLD, X_BOLD, Q_BOLD, V_BOLD , F_BOLD , flag= timeIntegrationRK4initNumba(dt, currentChunkSize, idxLastT,
                              N, SC, DM, maxDelay,
                              alpha, beta, gamma, delta, epsilon, tau,
                              I, K, sigma, max_bold,
                              noise, # c is already included in DM
                              xs, ys, dif,
                              BOLD,X_BOLD,Q_BOLD,F_BOLD,V_BOLD,
                              rho_b, alpha_b, V0_b, k1_b, k2_b, k3_b, Gamma_b, K_b, Tau_b, samplingRate_NDt,
                              min_xs, max_xs)


        if flag is False:
            return None
      
        if maxDelay == 0:
            xs[:,0] = xs[:,-1]
            ys[:,0] = ys[:,-1]
        else:  
            xs[:,:int(maxDelay)] = xs[:,-int(maxDelay):]
            ys[:,:int(maxDelay)] = ys[:,-int(maxDelay):]


        idxLastT = idxLastT + xs.shape[1]-maxDelay
    return BOLD[:,50:]





@numba.njit(locals = {'idxX': numba.int64, 'idxY':numba.int64, 'idx1':numba.int64, 'idy1':numba.int64})
def timeIntegrationRK4initNumba(dt, duration, idxLastT,
                              N, SC, DM, maxDelay,
                              alpha, beta, gamma, delta, epsilon, tau,
                              I, K, sigma, max_bold, # c is already included in DM
                              noise,
                              xs, ys, dif,
                              BOLD,X,Q,F,V,
                              rho_b, alpha_b, V0_b, k1_b, k2_b, k3_b, Gamma_b, K_b, Tau_b, samplingRate_NDt,
                              min_xs, max_xs):
    # load initial values


    x = xs[:,maxDelay-1].copy()
    y = ys[:,maxDelay-1].copy()

    for t in range(maxDelay, duration):  # start from max delay here - only consider interesting time steps!
        for n in range(N):             # all nodes
            

            x_ext = 0  # no y_ext since sum in FHN is only in u term
            for i in range(N):         # get input of every other node
                if dif==True:
                    x_ext = x_ext + SC[i, n] * (xs[i, int(np.round(t-DM[i,n]))]-x[n]) # if useDM false -> DM=0 -> doesnt matter
                else: 
                    x_ext = x_ext + SC[i, n] * xs[i, int(np.round(t-DM[i,n]))]  # transmission speed kappa (here: c) already in DM (s.o.)

            #x_input[n,t] = K * x_ext # + sigma * noise[n,t]
            # update FHN equations
            x_k1 = - alpha * x[n]**3 + beta * x[n]**2 + gamma * x[n] - y[n] + K * x_ext + sigma * noise[n,t] + I
            y_k1 = (x[n] - delta - epsilon*y[n])/tau
            x_k2 = - alpha * (x[n]+0.5*dt*x_k1)**3 + beta * (x[n]+0.5*dt*x_k1)**2 + gamma * (x[n]+0.5*dt*x_k1) - (y[n]+0.5*dt*y_k1) + K * x_ext + sigma * noise[n,t] + I
            y_k2 = ((x[n]+0.5*dt*x_k1) - delta - epsilon*(y[n]+0.5*dt*y_k1))/tau
            x_k3 = - alpha * (x[n]+0.5*dt*x_k2)**3 + beta * (x[n]+0.5*dt*x_k2)**2 + gamma * (x[n]+0.5*dt*x_k2) - (y[n]+0.5*dt*y_k2) + K * x_ext + sigma * noise[n,t] + I
            y_k3 = ((x[n]+0.5*dt*x_k2) - delta - epsilon*(y[n]+0.5*dt*y_k2))/tau
            x_k4 = - alpha * (x[n]+1.0*dt*x_k3)**3 + beta * (x[n]+1.0*dt*x_k3)**2 + gamma * (x[n]+1.0*dt*x_k3) - (y[n]+1.0*dt*y_k3) + K * x_ext + sigma * noise[n,t] + I
            y_k4 = ((x[n]+1.0*dt*x_k3) - delta - epsilon*(y[n]+1.0*dt*y_k3))/tau
            

            ### update x_n
            x[n] = x[n] + 1./6.*(x_k1+2*x_k2+2*x_k3+x_k4) * dt
            y[n] = y[n] + 1./6.*(y_k1+2*y_k2+2*y_k3+y_k4) * dt

            xs[n,t] = x[n]
            ys[n,t] = y[n]

            j=n
            
            
            X[j] = X[j] + dt*1e-3 * ( ((max_bold/(max_xs-min_xs))*(xs[j,t]-min_xs)) - K_b[j] * X[j] - Gamma_b[j] * (F[j] - 1) )
            Q[j] = Q[j] + dt*1e-3 / Tau_b[j] * ( F[j] / rho_b * (1- (1- rho_b)**(1/F[j])) \
                                            - Q[j] * V[j] **(1/alpha_b - 1 ) )
            V[j] = V[j] + dt*1e-3 / Tau_b[j] * ( F[j] - V[j] ** (1/alpha_b) )
            F[j] = F[j] + dt*1e-3 * X[j]
            



            if ((t+idxLastT - maxDelay)  % samplingRate_NDt) == 0:
                BOLD[j,(t+idxLastT-maxDelay)//samplingRate_NDt] = V0_b * ( k1_b * (1-Q[j]) + k2_b * (1-Q[j]/V[j]) + k3_b * (1-V[j]))
    
            if(np.isnan(x[n]) or np.isnan(y[n]) or np.isnan(X[j]) or np.isnan(Q[j])):
                return xs,ys, BOLD, X, Q, V, F, False     
    return xs,ys, BOLD, X, Q, V, F, True


@numba.njit
def get_min_max(dt, duration, 
                              N, SC, DM, maxDelay,
                              alpha, beta, gamma, delta, epsilon, tau,
                              I, K , sigma, 
                              noise, # c is already included in DM
                              xs, ys, dif):
    # load initial values

    x = xs[:,0].copy()
    y = ys[:,0].copy()
    
    for t in range(maxDelay, len(xs[0])):  # start from max delay here - only consider interesting time steps!
        
        for n in range(N):             # all nodes
            #load parameters
            

            x_ext = 0  # no y_ext since sum in FHN is only in u term
            for i in range(N):         # get input of every other node
                if dif==True:
                    x_ext = x_ext + SC[i, n] * (xs[i, int(np.round(t-DM[i,n]))]-x[n]) # if useDM false -> DM=0 -> doesnt matter
                else: 
                    x_ext = x_ext + SC[i, n] * xs[i, int(np.round(t-DM[i,n]))]  # transmission speed kappa (here: c) already in DM (s.o.)
            
            # update FHN equations
            x_k1 = - alpha * x[n]**3 + beta * x[n]**2 + gamma * x[n] - y[n] + K * x_ext + sigma * noise[n,t] + I
            y_k1 = (x[n] - delta - epsilon*y[n])/tau
            x_k2 = - alpha * (x[n]+0.5*dt*x_k1)**3 + beta * (x[n]+0.5*dt*x_k1)**2 + gamma * (x[n]+0.5*dt*x_k1) - (y[n]+0.5*dt*y_k1) + K * x_ext + sigma * noise[n,t] + I
            y_k2 = ((x[n]+0.5*dt*x_k1) - delta - epsilon*(y[n]+0.5*dt*y_k1))/tau
            x_k3 = - alpha * (x[n]+0.5*dt*x_k2)**3 + beta * (x[n]+0.5*dt*x_k2)**2 + gamma * (x[n]+0.5*dt*x_k2) - (y[n]+0.5*dt*y_k2) + K * x_ext + sigma * noise[n,t] + I
            y_k3 = ((x[n]+0.5*dt*x_k2) - delta - epsilon*(y[n]+0.5*dt*y_k2))/tau
            x_k4 = - alpha * (x[n]+1.0*dt*x_k3)**3 + beta * (x[n]+1.0*dt*x_k3)**2 + gamma * (x[n]+1.0*dt*x_k3) - (y[n]+1.0*dt*y_k3) + K * x_ext + sigma * noise[n,t] + I
            y_k4 = ((x[n]+1.0*dt*x_k3) - delta - epsilon*(y[n]+1.0*dt*y_k3))/tau
            
            ### update x_n
            x[n] = x[n] + 1./6.*(x_k1+2*x_k2+2*x_k3+x_k4) * dt 
            y[n] = y[n] + 1./6.*(y_k1+2*y_k2+2*y_k3+y_k4) * dt 

            ### save state
            xs[n,t] = x[n]
            ys[n,t] = y[n]

            
    
    return xs, ys, np.amin(xs[:,maxDelay:]), np.amax(xs[:,maxDelay:])


@numba.njit
def ou_x(runtime, dt, tau, mean, sigma_stat, rands, X0):
    '''
    generate OU process. [cf. https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process]
    parameters: tau, mean, sig_stat [sig_stat is the std of the stationary OU process]
    simulating the ou process according to the Langevin Eq.:
    dX = 1/tau*(mu - X)*dt + sigL * dW(t).
    sigL = sigma_stat*sqrt(2/tau) '''

    x = np.zeros(runtime+1)
    x[0] = X0
    # optimizations
    sigL = sigma_stat*np.sqrt(2./tau)
    sigL_sqrt_dt = sigL * np.sqrt(dt)
    dt_tau = dt/tau
    for i in range(runtime+1):
        x[i+1] = x[i] + dt_tau * (mean - x[i]) + sigL_sqrt_dt * rands[i]
    return x[1:]
    
def computeDelayMatrixdt(lengthMat,signalV,dt,segmentLength=1):
    """Compute the delay matrix from the fiber length matrix and the signal velocity
        
        :param lengthMat:       A matrix containing the connection length in segment
        :param signalV:         Signal velocity in m/s
        :param segmentLength:   Length of a single segment in mm  
      
        :returns:    A matrix of connexion delay in ms
    """
    """
    if signalV_cor and signalV_sub > 0:
        Dmat[0:41,0:41] = Dmat[0:41,0:41]/signalV_cor
        Dmat[47:75,47:75] = Dmat[47:75,47:75]/signalV_cor
        Dmat[82:94,82:94] = Dmat[82:94,82:94]/signalV_cor

        Dmat[41:46,41:46] = Dmat[41:46,41:46]/signalV_sub
        Dmat[75:82,75:82] = Dmat[75:82,75:82]/signalV_sub
    """


    normalizedLenMat = lengthMat * segmentLength    # Each segment is ~1.8mm
    if signalV > 0:
        Dmat = np.ceil(normalizedLenMat / signalV / dt)  # Interareal connection delays, Dmat(i,j) in ms
    else:
        Dmat = lengthMat * 0.0
    return Dmat.astype(int)


def load_parameters(singleNode=False, globalN=0):
    class struct(object):
        pass

    params = struct()
    #FHN parameters with:
    '''
    du/dt = -alpha u^3 + beta u^2 - gamma u - w + I_{ext}
    dw/dt = 1/tau (u + delta  - epsilon w)
    '''
    params.alpha = 3. # eps in "kostova2005" paper
    params.beta = 4. # eps(1+lam)
    params.gamma = -1.5 # lam eps
    params.delta = 0.
    params.epsilon = 0.5 # a
    params.tau = 20. 

    
    ### runtime parameters
    params.dt = 0.1  # simulation timestep in ms 
    params.duration = 300000  # Simulation duration in ms

    ### network parameters
    """
    if singleNode: # if you only want a single node with those parameters
        N = 1
        params.SC = np.zeros((N, N))
        params.DM = np.zeros((N, N))
    elif globalN>0: # if you want a network of N nodes, all uniformly connected
        N = globalN
        SC = np.ones((N,N))
        np.fill_diagonal(SC, 0)
        params.SC = SC

    else: # this is what you usually want! Here you load the netowrk topology from the DTI images
        SCs90_HCP10 = np.load(dire)
        SC = np.mean(SCs90_HCP10,axis=0)
        
       

        N = SC.shape[0]          
        params.SC = SC
    
    params.N = N  # number of nodes
    """
    ### global parameters



    params.I = 0.5 # Background input
    params.K = 0.  # global coupling strength
    params.sigma = 1 # Variance of the additive noise
    params.c = 1  # signal transmission speed in ms
    
    ### Initialization noise
    #params.init = (0,1,2,3) #initialize randomly with u in [0,1] & w in [2,3]
    params.xinit = np.zeros(80)
    params.yinit = np.zeros(80)

    params_dict = params.__dict__
    return params_dict



def matrix_correlation(M1, M2):
    if M1 is None or M2 is None:
        return 0
    return np.corrcoef(M1.reshape((1,M1.size)), M2.reshape( (1,M2.size) ) )[0,1]
    
def fc(BOLD):
    simFC = np.corrcoef(BOLD)
    simFC = np.nan_to_num(simFC) # remove NaNs
    return simFC




def kolmogorov(BOLD1, BOLD2, windowsize = 1.0):
    # return kolmogorov distance between two functional connectivities
    empiricalFCD = fcd(BOLD2[:,:len(BOLD1[0,:])], windowsize = windowsize)
    FCD = fcd(BOLD1, windowsize = windowsize);
    
    triUFCD = np.triu(FCD)
    triUFCD = triUFCD[(triUFCD>0.0)&(triUFCD<1.0)]

    emptriUFCD = np.triu(empiricalFCD)
    emptriUFCD = emptriUFCD[(emptriUFCD>0.0)&(emptriUFCD<1.0)]

    return scipy.stats.ks_2samp(triUFCD, emptriUFCD)[0]
def fcd(BOLD, windowsize = 30, stepsize = 5, N=90):
    # compute FCD matrix
    t_window_width = int(windowsize)# int(windowsize * 30) # x minutes
    stepsize = stepsize # BOLD.shape[1]/N
    corrFCs = []
    try:
        counter = range(0, BOLD.shape[1]-t_window_width, stepsize)
        
        for t in counter:
            BOLD_slice = BOLD[:,t:t+t_window_width]
            corrFCs.append(np.corrcoef(BOLD_slice))
        
        FCd = np.empty([len(corrFCs),len(corrFCs)])
        f1i = 0
        
        for f1 in corrFCs:
            f2i = 0
            for f2 in corrFCs:
                FCd[f1i, f2i] = np.corrcoef(f1.reshape((1,f1.size)), f2.reshape( (1,f2.size) ) )[0,1]
                f2i+=1
            f1i+=1
        return FCd
    except:
        return 0

def kolmogorov_fcd(empiricalFCD, FCD): 
    triUFCD = np.triu(FCD)
    triUFCD = triUFCD[(triUFCD>0.0)&(triUFCD<1.0)]

    emptriUFCD = np.triu(empiricalFCD)
    emptriUFCD = emptriUFCD[(emptriUFCD>0.0)&(emptriUFCD<1.0)]

    return scipy.stats.ks_2samp(triUFCD, emptriUFCD)[0]