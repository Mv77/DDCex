# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Solving a Dynamic Discrete Choice Problem with Three Different Methods
#
# # Mateo Velásquez-Giraldo
#
# This notebook solves a simple dynamic "machine replacement" problem using three different methods:
# - Contraction mapping iteration.
# - Hotz-Miller inversion.
# - Forward simulation.
#
# The code is optimized for clarity, not speed, as the purpose is to give a sense of how the three methods work and how they can be implemented.

# %%
# Setup
import numpy as np
from scipy.optimize import minimize, Bounds
import pandas as pd
import numpy.random as rnd
import copy

# %% [markdown]
# # Problem setup
#
# The problem is taken from a problem set of Prof. Nicholas Papageorge Topics in Microeconometrics course at Johns Hopkins University and borrows heavily from Professor Juan Pantano's course Microeconometrics offered at Washington University in St. Louis.
#
# There is a shop that operates using a machine. The machine's mantainance costs increase with its age, denoted $a_t$. On each period, the shop must decide whether to replace the machine ($i_t = 1$) or not ($i_t=0$). Assume that costs stop increasing after the machine reaches $a_t = 5$ so that, in practice, that is the maximum age. Age then evolves according to:
# \begin{equation}
# a_{t+1} = \begin{cases}
# 		\min \{5,a_t+1\}, & \text{ if } i_t = 0 \\
# 		1, & \text{ if } i_t = 1
# 		\end{cases}.
# \end{equation}
#
# A period's profits depend on mantainance costs, replacement costs, and factors that the econometrician does not observe, modeled as stochastic shocks $\epsilon$:
# \begin{equation}
#     \Pi (a_t,i_t,\epsilon_{0,t},\epsilon_{1,t}) = \begin{cases}
#         \theta a_t + \epsilon_{0,t} & \text{if } i_t=0\\
#         R + \epsilon_{1,t} &  \text{if } i_t = 1  
#     \end{cases}
# \end{equation}
#
# The shop's problem can be recursively defined as:
# \begin{equation}
# 	\begin{split}
# 		V(a_t,\epsilon_{0,t},\epsilon_{1,t}) &= \max_{i_t} \Pi 
# 		(a_t,i_t,\epsilon_{0,t},\epsilon_{1,t}) + \beta 
# 		E_t[V(a_{t+1},\epsilon_{0,t+1},\epsilon_{1,t+1})]\\
# 		&\text{s.t} \\
# 		a_{t+1} &= \begin{cases}
# 		\min \{5,a_t+1\}, & \text{ if } i_t = 0 \\
# 		1, & \text{ if } i_t = 1
# 		\end{cases}.
# 	\end{split}
# \end{equation}
#
# The code below defines functions and objects that capture the structure of the problem

# %%
# Profit function (the deterministic part)
def profit_det(a,i,theta,R):
    
    if i == 0:
        return(theta*a)
    else:
        return(R)

# State transition function
def transition(a, i):
    if i == 0:
        return(min(5,a+1))
    else:
        return(1)

    
# Construct state and choice vectors
states = np.arange(5) + 1
choices = np.arange(2)

# Construct the transition matrix array:
# A 2 x 5 x 5 array in which the position (i,j,k) contains
# the probability of moving from state j to state k given that
# choice i was made

trans_mat = np.zeros((len(choices),len(states),len(states)))

# If no-replacement, deterministically move to the next state, up to the last
for k in range(len(states)-1):
    trans_mat[0][k][k+1] = 1
trans_mat[0,len(states)-1,len(states)-1] = 1

# If replacement, deterministically move to the first state
for k in range(len(states)):
    trans_mat[1,k,0] = 1


# %% [markdown]
# ## Some more notation
#
# The solution methods use objects that are derived from the value function $V$ and that will be defined below.
#
# ### Pre-Shocks Expected value function $\tilde{V}(\cdot)$
#
# This object captures the lifetime utility a shop can expect after knowing its state $a_t$ but before knowing its stochastic shock realizations.
#
# \begin{equation}
#     \tilde{V}(a_t) = E_\epsilon [V(a_t,\epsilon_{0,t},\epsilon_{1,t})]
# \end{equation}
#
# ### Choice-Specific Value Functions $\bar{V}_{i}(\cdot)$
#
# These two objects capture the lifetime utility expected from a choice, excluding the current-period stochastic shock. Formally, they are:
#
# \begin{equation}
# 	\bar{V}_0(a_t) = \theta_1 a_t + \beta E \left[ V(\min\left\{ 5, a_t+1\right\},\epsilon_{0,t+1},\epsilon_{1,t+1}\right)]
# \end{equation}
#
# and
#
# \begin{equation}
# \bar{V}_1(a_t) = R + \beta E \left[ V(1,\epsilon_{0,t+1},\epsilon_{1,t+1}\right)].
# \end{equation}
#
#
# ## Useful relationships
#
# The previously defined objects are related through the following identities
#
# \begin{equation}
#     \bar{V}_i\left( a_t \right) = \Pi (a_t,i_t,0,0) + \beta\tilde{V}\left(a_{t+1}\left(a_t,i\right)\right),
# \end{equation}
# and
# \begin{equation}
# 	V(a_t,\epsilon_{0,t},\epsilon_{1,t}) = \max \left\{ \bar{V}_0\left( 
# 	a_t \right) + \epsilon_{0,t}, \bar{V}_1\left( 
# 	a_t \right) + \epsilon_{1,t} \right\}.
# \end{equation}
#
# ## Choice probabilities
#
# Using the last relationship and assuming that a shop behaves optimally, it should be the case that
#
# \begin{equation}
#     i_t = \arg \max_{i\in \{0,1\}} \left( \bar{V}_i\left( a_t \right) + \epsilon_{i,t}\right).
# \end{equation}
#
# Assuming that stochastic shocks $\epsilon$ are i.i.d Extreme-value-type-1 yields a simple expression for the probability of choosing each alternative:
#
# \begin{equation}
# P(i_t=1|a_t) = \frac{\exp (\bar{V}_1(a_t))}{\exp (\bar{V}_0(a_t))+\exp (\bar{V}_1(a_t))}.
# \end{equation}
#
# This expression allows us to estimate the model's parameters given data through maximum likelihood estimation. The likelihood function would be
#
# \begin{equation}
# \mathcal{L}(\theta,R) = \Pi_{j=1}^N P\left( i_j|a_j,\theta, R\right).
# \end{equation}
#
# We now only need ways to obtain choice-specific net-of-error value functions $\bar{V}_i(\cdot)$ for any given set of parameters. In this notebook we will explore three.

# %%
# Compute the log-likelihood of (a,i) vectors given choice-specific,
# net-of-error value functions
def logL(a, i, V):
    
    # Compute the probability of each (a,i) pair possible
    probs = np.exp(V)
    total = np.sum(probs, axis = 1)
    probs = probs / total[:,None]
    
    # Get a vector of the probabilities of observations
    L = probs[a-1,i]
    logLik = np.sum(np.log(L))
    return(logLik)


# %% [markdown]
# # Solution of the dynamic problem
#
# To simulate data, we must first solve the problem. We must then introduce the first method that we will use.

# %% [markdown]
# ## 1. Contraction mapping iteration
#
# A first way of obtaining choice-specific value functions is defining the following mapping.
#
# \begin{equation}
#     T\left(\begin{bmatrix}
#     f_0(\cdot)\\
#     f_1(\cdot)
#     \end{bmatrix}\right)(a_t) = \begin{bmatrix}
#     \theta_1 a_t + \beta E [\max \left\{ f_0\left( a_{t+1}\left(a_t,i_t=0\right)\right) + \epsilon_{0,t}, f_1\left( \left( a_{t+1}\left(a_t,i_t=0\right) \right) \right) + \epsilon_{1,t} \right\}] \\
#     R + \beta E [ \max \left\{ f_0\left( 
# 	\left( a_{t+1}\left(a_t,i_t=1\right) \right) \right) + \epsilon_{0,t}, f_1\left( 
# 	\left( a_{t+1}\left(a_t,i_t=1\right) \right) \right) + \epsilon_{1,t} \right\}]
#     \end{bmatrix}
# \end{equation}
#
# and noting that $[\bar{V}_0(\cdot),\bar{V}_1(\cdot)]'$ is a fixed point of $T$.
#
# In fact, $T$ is a contraction mapping, so a strategy for finding its fixed point is iteratively applying $T$ from an arbitrary starting point. This is precisely what the code below does.

# %%
# Computation of E[max{V_0 + e0, V_1 + e1}]
def expectedMax(V0,V1):
    return( np.euler_gamma + np.log( np.exp(V0) + np.exp(V1) ) )

# Contraction mapping
def contrMapping(Vb, theta, R, beta):
    
    # Initialize array (rows are a, cols are i)
    Vb_1 = np.zeros( Vb.shape )
    
    for a_ind in range(len(Vb)):
        
        # Adjust 0 indexing
        a = a_ind + 1
        
        for i in range(2):
            
            a_1 = transition(a, i)
            a_1_ind = a_1 - 1
            Vb_1[a_ind, i] = profit_det(a, i, theta, R) + \
                            beta * expectedMax(Vb[a_1_ind,0],Vb[a_1_ind,1])
        
    return(Vb_1)

# Solution of the fixed point problem by repeated application of the
# contraction mapping
def findFX(V0, theta, R, beta, tol, disp = True):
    
    V1 = V0
    norm = tol + 1
    count = 0
    while norm > tol:
        count = count + 1
        V1 = contrMapping(V0, theta, R, beta)
        norm = np.linalg.norm(V1 - V0)
        if disp:
            print('Iter. %i --- Norm of difference is %.6f' % (count,norm))
        V0 = V1
        
    return(V1)


# %% [markdown]
# ## 2. Hotz-Miller Inversion
#
# The Hotz-Miller method relies on the following re-expression of the pre-shock expected value function
#
# \begin{equation}
#     \tilde{V}(a_t) = \sum_{i\in\{0,1\}} P(i_t = i | a_t) \times \left( \Pi \left(a_t,i_t,0,0\right) + E\left[ \epsilon_i | i_t = i\right] + \sum_{a'= 1}^{5} P\left(a_{t+1} = a' | a_t, i_t = i\right) \tilde{V}\left(a'\right) \right)
# \end{equation}
#
# which is a system of linear equations in $\{ \tilde{V}(1),...,\tilde{V}(5) \}$ if one knows $ P(i_t = i | a_t)$, $\Pi\left(a_t,i_t,0,0\right)$, $E\left[ \epsilon_i | i_t = i\right]$, and $P\left(a_{t+1} = a' | a_t, i_t = i\right)$.
#
# - $ P(i_t = i | a_t)$ are known as "conditional choice probabilities", and can be estimated from the data directly.
#
# - $P\left(a_{t+1} = a' | a_t, i_t = i\right)$ are state-to-state transition probabilities. In our simple problem, transitions are deterministic, but in more complex problems these could also be directly estimated from the data.
#
# - $\Pi\left(a_t,i_t,0,0\right)$ is known given parameters.
#
# - $E\left[ \epsilon_i | i_t = i\right]$ is equal to $\gamma - \ln P(i_t = i|a_t)$ if one assumes i.i.d extreme value type one errors ($\gamma$ is Euler's constant).
#
# Thus, for any given parameter vector we can solve the linear system for $\{ \tilde{V}(1),...,\tilde{V}(5) \}$. With these, we can use the previously defined relationship
#
# \begin{equation}
#     \bar{V}_i\left( a_t \right) = \Pi (a_t,i_t,0,0) + \beta\tilde{V}\left(a_{t+1}\left(a_t,i\right)\right),
# \end{equation}
#
# to obtain choice-specific, net-of-error value functions and obtain our likelihood.

# %%
def Hotz_Miller(theta, R, states, choices, CPPS, trans_mat,invB):
    
    nstates = len(states)
    nchoices = len(choices)
    
    # Construct ZE matrix
    ZE = np.zeros((nstates, nchoices))
    for i in range(nstates):
        for j in range(nchoices):
            ZE[i,j] = CPPS[i,j]*( profit_det(states[i],choices[j],theta,R) +
                                  np.euler_gamma - np.log(CPPS[i,j]) )
    
    # Take a sum.
    ZE = np.sum(ZE,1,keepdims = True)
    # Compute W
    W = np.matmul(invB, ZE)
    
    # Z and V
    Z = np.zeros((nstates,nchoices))
    V = np.zeros((nstates,nchoices))
    for i in range(nstates):
        for j in range(nchoices):
            Z[i,j] = np.dot(trans_mat[j][i,:],W)
            
            V[i,j] = profit_det(states[i],choices[j],theta,R) + beta*Z[i,j]
    return(V)


# %% [markdown]
# ## 3. Forward Simulation

# %%
def forward_simul(theta,R,beta,states,choices,CPPS,trans_mat,nperiods,nsims,
                  seed):
    
    # Set seed
    rnd.seed(seed)
    
    # Initialize V
    V = np.zeros((len(states),len(choices)))
    
    for i in range(len(states)):
        for j in range(len(choices)):
            
            v_accum = 0
            for r in range(nsims):
                
                a_ind = i
                c_ind = j
                v = profit_det(states[a_ind], choices[c_ind], theta, R)
                
                for t in range(nperiods):
                    
                    # Simulate state
                    a_ind = rnd.choice(a = len(states),
                                            p = trans_mat[c_ind][a_ind])
                    
                    # Simulate choice
                    c_ind = rnd.choice(a = len(choices),
                                            p = CPPS[a_ind])
                    
                    # Find expected value of taste disturbance conditional on
                    # choice
                    exp_e = np.euler_gamma - np.log(CPPS[a_ind,c_ind])
                    # Update value funct
                    v = v + ( beta**(t+1) ) * (profit_det(states[a_ind],
                                                          choices[c_ind],
                                                          theta,R) +
                                               exp_e)

                v_accum = v_accum + v
            V[i,j] = v_accum / nsims
    
    return(V)



# %% [markdown]
# # Dataset simulation
#
# Now, to simulate the model, we only need to solve the problem for some set of parameters and, using the result and simulated taste shocks, produce optimal behavior.
#
# The function below does exactly this, simulating a panel of machines, each observed for some pre-set number of periods.

# %%
def sim_dataset(theta, R, nmachines, n_per_machine, beta):

    # First solve the choice specific value functions for both parameter sets
    V0 = np.zeros((5,2))
    tol = 1e-6 # Tolerance
    
    V = findFX(V0, theta, R, beta, tol, disp = False)
    
    data = pd.DataFrame(np.zeros((nmachines*n_per_machine,4)),
                        columns = ['Id','T','a','i'])
    
    ind = 0
    for m in range(nmachines):
        
        # Initialize state
        a_next = rnd.randint(5) + 1
        
        for t in range(n_per_machine):
            
            a = a_next
            
            # Assign id and time
            data.loc[ind,'Id'] = m
            data.loc[ind, 'T'] = t
            
            data.loc[ind, 'a'] = a
            
            u_replace = V[a - 1][1] + rnd.gumbel()
            u_not     = V[a - 1][0] + rnd.gumbel()
            
            if u_replace < u_not:
                data.loc[ind,'i'] = 0
                a_next = min(5, a+1)
            else:
                data.loc[ind,'i'] = 1
                a_next = 1
                
            ind = ind + 1
            
    return(data)


# %% [markdown]
# Now we can use the function to simulate a full dataset.

# %%
# Simulate a dataset of a single type
nmachines = 6000
n_per_machine = 1

# Assign test parameters
theta = -1
R = -4
beta = 0.85

data = sim_dataset(theta, R, nmachines, n_per_machine, beta)
a = data.a.values.astype(int)
i = data.i.values.astype(int)


# %% [markdown]
# It is also useful to define functions that estimate conditional choice probabilities and state-to-state transition probabilities from the data, since we will be using them in estimation for some methods.

# %%
def get_ccps(states, choices):
    # Function to estimate ccps. Since we are in a discrete setting,
    # these are just frequencies.
    
    # Find unique states
    un_states = np.unique(states)
    un_states.sort()
    un_choices = np.unique(choices)
    un_choices.sort()
    
    # Initialize ccp matrix
    ccps = np.ndarray((len(un_states),len(un_choices)), dtype = float)
    
    # Fill out the matrix
    for i in range(len(un_states)):
        
        sc = choices[states == un_states[i]]
        nobs = len(sc)
        
        for j in range(len(un_choices)):
            
            ccps[i][j] = np.count_nonzero( sc == un_choices[j]) / nobs
    
    return(ccps)

def state_state_mat(CPP,transition_mat):
    
    nstates = CPP.shape[0]
    nchoices = CPP.shape[1]
    # Initialize
    PF = np.zeros((nstates,nstates))
    
    for i in range(nstates):
        for j in range(nstates):
            for d in range(nchoices):
                PF[i,j] = PF[i,j] + CPP[i,d]*transition_mat[d][i,j]

    return(PF)


# %% [markdown]
# Now we use the functions to estimate the CCPS and the transition matrix in the dataset that we just simulated.

# %%
# Estimate CPPS
cpps = get_ccps(a,i)

# Compute the state-to-state (no choice matrix)
PF = state_state_mat(cpps,trans_mat)


# %% [markdown]
# # Estimation
#
# We are now ready to estimate the model using our data and the three methods that were previously discussed.
#
# In every case, we define a function that takes the parameters and data, solves the model using the specific method, and computes the log-likelihood. All that is left then is to optimize!

# %% [markdown]
# ## 1. Rust's contraction mapping.

# %%
# Compute the log-likelihood of (a,i) vectors given parameter values,
# with contraction mapping method    
def logL_par_fx(par, a, i, tol):
    
    # Extract parameters
    theta = par[0]
    R = par[1]
    beta = par[2]
    
    # Find implied value functions
    V = np.zeros((5,2))
    V = findFX(V, theta, R, beta, tol, disp = False)
    
    # Return the loglikelihood from the implied value function
    return(logL(a, i, V) )
# %%
# Set up the objective function for minimization
tol  = 1e-9
x0   = np.array([0,0]) 
obj_fun_fx = lambda x: -1 * logL_par_fx([x[0],x[1],beta], a, i, tol)

# Optimize
est_fx = minimize(obj_fun_fx, x0, method='BFGS', options={'disp': True})
mean_est_fx = est_fx.x

se_est_fx = np.diag(est_fx.hess_inv)

# %%
# Present results
print('Estimation results (S.E\'s in parentheses):')
print('Theta: %.4f (%.4f)' % (mean_est_fx[0], se_est_fx[0]))
print('R: %.4f (%.4f)' % (mean_est_fx[1], se_est_fx[1]))


# %% [markdown]
# ## 2. Hotz-Miller

# %%
# Compute the log-likelihood of (a,i) vectors given parameter values,
# with forward simulation method
def logL_par_HM(par, a, i,
                states, choices, CPPS, trans_mat, 
                invB):
    
    # Extract parameters
    theta = par[0]
    R = par[1]
    
    # Find implied value functions
    V = Hotz_Miller(theta, R, states, choices, CPPS, trans_mat,invB)
    
    # Return the loglikelihood from the implied value function
    return(logL(a, i, V) )
# %%
# Compute the "inv B" matrix
invB = np.linalg.inv( np.identity(len(states)) - beta*PF )

# Set up objective function
obj_fun_HM = lambda x: -1 * logL_par_HM(x, a, i,states, choices,
                                        cpps, trans_mat, invB)

# Optimize
est_HM = minimize(obj_fun_HM, x0, method='BFGS', options={'disp': True})
mean_est_HM = est_HM.x

se_est_HM = np.diag(est_HM.hess_inv)

# %%
# Present results
print('Estimation results (S.E\'s in parentheses):')
print('Theta: %.4f (%.4f)' % (mean_est_HM[0], se_est_HM[0]))
print('R: %.4f (%.4f)' % (mean_est_HM[1], se_est_HM[1]))


# %% Pre-define important structures [markdown]
# ## 3. Forward Simulation

# %% Estimate using contraction mapping
# Compute the log-likelihood of (a,i) vectors given parameter values,
# with forward simulation method
def logL_par_fs(par, a, i,
                states, choices, CPPS, trans_mat, 
                nperiods, nsims, seed):
    
    # Extract parameters
    theta = par[0]
    R = par[1]
    beta = par[2]
    
    # Find implied value functions
    V = forward_simul(theta,R,beta,
                      states,choices,
                      CPPS,trans_mat,
                      nperiods,nsims,
                      seed)
    # Return the loglikelihood from the implied value function
    return(logL(a, i, V) )


# %% Estimate using Hotz-Miller
nperiods = 40
nsims    = 30
seed     = 1

# Set up objective function
obj_fun_fs = lambda x: -1 * logL_par_fs([x[0],x[1],beta],a,i,
                                        states, choices, cpps, trans_mat,
                                        nperiods = nperiods, nsims = nsims,
                                        seed = seed)

# Optimize
est_fs = minimize(obj_fun_fs, x0, method='BFGS', options={'disp': True})
mean_est_fs = est_fs.x

se_est_fs = np.diag(est_fs.hess_inv)

# %%
# Present results
print('Estimation results (S.E\'s in parentheses):')
print('Theta: %.4f (%.4f)' % (mean_est_fs[0], se_est_fs[0]))
print('R: %.4f (%.4f)' % (mean_est_fs[1], se_est_fs[1]))
