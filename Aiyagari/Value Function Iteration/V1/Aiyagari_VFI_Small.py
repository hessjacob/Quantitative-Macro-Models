"""
Author: Jacob Hess 
Date: January 2021

Written in python 3.8 on Spyder IDE.

Description: Finds the stationary equilibrium in a production economy with incomplete markets and idiosyncratic income
risk as in Aiyagari (1994). Features of the algorith are 1) two income states and a transition matrix eoxgenously set
2) value function iteration to solve the household problem 3) approximation of the stationary distribution using 
a monte carlo simulation

Aknowledgements: I used notes or pieces of code from the following :
    1) Gianluca Violante's notes (https://sites.google.com/a/nyu.edu/glviolante/teaching/quantmacro)
    2) Fabio Stohler (https://github.com/Fabio-Stohler)
    3) Jeppe Druedahl (https://github.com/JeppeDruedahl) and NumEconCopenhagen (https://github.com/NumEconCopenhagen)
    
Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual)
    -- Interpolation from EconForge
       * optimized interpolation routines for python/numba
       * to install 'conda install -c conda-forge interpolation'
       * https://github.com/EconForge/interpolation.py
    
"""


import time
import numpy as np
from numba import njit, prange
from interpolation import interp
import matplotlib.pyplot as plt
import seaborn as sns





#############
# I. Model  #
############

class AiyagariVFISmall:
    
    """
    Class object of the model. AiyagariVFISmall().solve_model() runs everything
    """

    ############
    # 1. setup #
    ############

    def __init__(self, sigma=2,               #crra coefficient
                       a_bar = 0,             #select borrowing limit
                       plott =1,              #select 1 to make plots
                       plot_supply_demand = 1 # select 1 for capital market supply/demand graph
                       ):
        
        #parameters subject to changes
        self.sigma, self.a_bar, self.plott, self.plot_supply_demand  = sigma, a_bar, plott, plot_supply_demand
        
        self.setup_parameters()
        self.setup_grid()
        

    def setup_parameters(self):

        # a. model parameters
        self.beta = 0.96  # discount factor
        self.rho = (1-self.beta)/self.beta #discount rate
        self.delta = 0.08  # depreciation rate
        self.alpha = 0.36  # cobb-douglas coeffient

        # b. hh solution
        self.tol_hh = 1e-6  # tolerance for consumption function iterations
        self.maxit_hh = 2000  # maximum number of iterations when finding consumption function in hh problem
       
        # income
        self.Nz = 2
        self.grid_z = np.array([0.5, 1.5])                #productuvity states
        self.pi = np.array([[3/4, 1/4],[1/4, 3/4]])   #transition probabilities

        # asset grid 
        self.Na = 1000
        self.a_min = self.a_bar
        self.a_max = 40
        self.curv = 3 


        # c. simulation
        self.seed = 123
        self.ss_a0 = 1.0  # initial cash-on-hand (homogenous)
        self.ss_simN = 50_000  # number of households
        self.ss_simT = 2000  # number of time-periods
        self.ss_sim_burnin = 1000  # burn-in periods before calculating average savings

        # d. steady state solution
        self.ss_ret_tol = 1e-4  # tolerance for finding interest rate
        self.dp_big = 1/10      # dampening parameter to update new interest rate guess 
        self.dp_small = 1/100    # dampening parameter to prevent divergence
        self.maxit = 100    # maximum iterations steady state solution
        
        # e. complete markets solution
        self.ret_cm = 1/self.beta - 1
        self.k_cm = self.k_demand(self.ret_cm)


    def setup_grid(self):
        # a. asset grid
        self.grid_a = self.make_grid(self.a_min, self.a_max, self.Na, self.curv)  #savings grid

        # b. initial distribution of z
        z_diag = np.diag(self.pi ** 1000)
        self.ini_p_z = z_diag / np.sum(z_diag)

        # c. income grid
        avg_z = np.sum(self.grid_z * self.ini_p_z)
        self.grid_z = self.grid_z / avg_z  # force mean one

        
        
    
    
    
    
    
    
    
    
    
    
    #######################
    # 2. helper functions #
    ######################
    
    def make_grid(self, min_val, max_val, num, curv):  
        """
        Makes an exponential grid of degree curv. 
        
        A higher curv will put more points closer a_min. 
        
        Equivalenty, np.linspace(min_val**(1/curv), max_val**(1/curv), num)**curv will make
        the exact same grid that this function does.
        """
        grd = np.zeros(num)
        scale=max_val-min_val
        grd[0] = min_val
        grd[num-1] = max_val
        for i in range(1,num-1):
            grd[i] = min_val + scale*((i)/(num - 1)) ** curv
        
        return grd
    
    
    ##############
    # household #
    #############
    
    def u(self, c):
        eps = 1e-8
        
        if  self.sigma == 1:
            return np.log(np.fmax(c, eps))
        else:
            return (np.fmax(c, eps) ** (1 - self.sigma) -1) / (1 - self.sigma)


    def u_prime(self, c) :
        eps = 1e-8
        
        return np.fmax(c, eps) ** (-self.sigma)

    
    def u_prime_inv(self, x):    
        eps = 1e-8
        
        return np.fmax(x, eps) ** (-1/self.sigma)
    
    #########
    # firm #
    ########
    
    def  f(self,k) :
        eps = 1e-8
        return np.fmax(k, eps) ** self.alpha
    
    def f_prime(self,k):
        eps = 1e-8
        return self.alpha * np.fmax(k, eps) ** (self.alpha - 1)
    
    def f_prime_inv(self,x):
        eps = 1e-8
        return (np.fmax(x, eps) / self.alpha) ** ( 1 / (self.alpha - 1) )
    
    def ret_func(self, k):
        return  self.f_prime(k) - self.delta

    def w_func(self, ret):
        k = self.f_prime_inv(ret + self.delta)
        return self.f(k) - self.f_prime(k) * k
    
    def k_demand(self,ret):
        return (self.alpha/(ret+self.delta))**(1/(1-self.alpha))
    
    
    


    
    
    #############################
    # 3. stationary equilibrium #
    #############################
    
    def graph_supply_demand(self,ret_vec):
        
        """
        Plots capital market supply and demand.
        
        *Input
            - ret_vec : vector of interest rates
        *Output
            - k_demand : capital demand as a function of the interest rate
            - k_supply : capital supply as a function of the interest rate
        """
        
        #1. initialize
        k_demand = np.empty(ret_vec.size)
        k_supply = np.empty(ret_vec.size)
        
        for idx, ret_graph in enumerate(ret_vec):
            
            # 2. capital demand
            k_demand[idx] = self.k_demand(ret_graph)
            
            # 3. capital supply
            w_graph = self.w_func(ret_graph)
            
            # i. Household problem
            VF_graph, pol_sav_graph, pol_cons_graph, it_hh_graph = solve_hh(ret_graph, self.Nz, self.Na, self.tol_hh, self.maxit_hh, 
                      self.grid_a, w_graph, self.grid_z, self.sigma, self.beta, self.pi)
            
            
            # ii. Simulation
            a0 = self.ss_a0 * np.ones(self.ss_simN)
            z0 = np.zeros(self.ss_simN, dtype=np.int32)
            z0[np.linspace(0, 1, self.ss_simN) > self.ini_p_z[0]] = 1
            sim_ret_graph = ret_graph * np.ones(self.ss_simT)
            sim_w_graph = w_graph * np.ones(self.ss_simT)
            
            sim_k_graph, a, z, c, m = simulate_MonteCarlo(
                a0,
                z0,
                sim_ret_graph,
                sim_w_graph,
                self.ss_simN,
                self.ss_simT,
                self.grid_z,
                self.grid_a,
                pol_cons_graph,
                pol_sav_graph,
                self.pi,
                self.seed,
            )
            
            k_supply[idx] = np.mean(sim_k_graph[self.ss_sim_burnin:])

        return k_demand, k_supply
            

    def ge_algorithm(self, ret_ss_guess, a0, z0, t1):
        
        """
        General equilibrium solution algorithm.
        """
        
        #given ret_ss_guess as the guess for the interest rate (step 1)
        
        # a. obtain prices from firm FOCs (step 2)
        self.ret_ss = ret_ss_guess
        self.w_ss = self.w_func(self.ret_ss)

        # b. solve the HH problem (step 3)
        
        print('\nSolving household problem...')
        
        self.VF, self.pol_sav, self.pol_cons, self.it_hh = solve_hh(self.ret_ss, self.Nz, self.Na, self.tol_hh, self.maxit_hh, 
                      self.grid_a, self.w_ss, self.grid_z, self.sigma, self.beta, self.pi)
        
        
        if self.it_hh > self.maxit_hh:
            raise Exception('No value function convergence')
        else : 
            print(f"Value function convergence in {self.it_hh} iterations.")

            
        t2 = time.time()
        print(f'Household problem time elapsed: {t2-t1:.2f} seconds')

        # c. simulate (step 4)
        
        print('\nSimulating...')
        
        # prices
        self.ss_sim_ret = self.ret_ss * np.ones(self.ss_simT)
        self.ss_sim_w = self.w_ss * np.ones(self.ss_simT)
        
        self.ss_sim_k, self.ss_sim_a, self.ss_sim_z, self.ss_sim_c, self.ss_sim_m = simulate_MonteCarlo(
            a0,
            z0,
            self.ss_sim_ret,
            self.ss_sim_w,
            self.ss_simN,
            self.ss_simT,
            self.grid_z,
            self.grid_a,
            self.pol_cons,
            self.pol_sav,
            self.pi,
            self.seed,
        )
        
        t3 = time.time()
        print(f'Simulation time elapsed: {t3-t2:.2f} seconds')

        # d. calculate difference
        self.k_ss = np.mean(self.ss_sim_k[self.ss_sim_burnin :])
        ret_ss_new = self.ret_func(self.k_ss)
        diff = ret_ss_guess - ret_ss_new
        
        return diff





    #####################
    # 4. Main Function #
    ####################

    def solve_model(self):
    
            """
            Finds the stationary equilibrium
            """    
            
            t0 = time.time()    #start the clock
    
            # a. initial values for agents
            a0 = self.ss_a0 * np.ones(self.ss_simN)
            z0 = np.zeros(self.ss_simN, dtype=np.int32)
            z0[np.linspace(0, 1, self.ss_simN) > self.ini_p_z[0]] = 1
    
            # b. initial interest rate guess (step 1)
            ret_guess = 0.02       
            
            # We need (1+r)beta < 1 for convergence.
            assert (1 + ret_guess) * self.beta < 1, "Stability condition violated."
            
            # c. iteration to find equilibrium interest rate ret_ss
            
            for it in range(self.maxit) :
                t1=time.time()
                
                print("\n-----------------------------------------")
                print("Iteration #"+str(it+1))
                
                diff_old=np.inf
                diff = self.ge_algorithm(ret_guess, a0, z0, t1)
                
                if abs(diff) < self.ss_ret_tol :
                    print('\nConvergence')
                    self.ret_ss = ret_guess
                    break
                else :
                    #adaptive dampening 
                    if diff > diff_old:
                        ret_guess = ret_guess - self.dp_small*diff  #to prevent divergence force a conservative new guess
                    else:
                        ret_guess = ret_guess - self.dp_big*diff
                    
                    print(f"\nNew interest rate guess = {ret_guess:.5f} \t diff = {diff:8.5f}")
                    diff_old=diff
            
            if it > self.maxit :
                print("No convergence")
                
            #calculate precautionary savings rate
            self.precaution_save = self.ret_cm - self.ret_ss
            
            t4 = time.time()
            print('Total iteration time elapsed: '+str(time.strftime("%M:%S",time.gmtime(t4-t0))))
            
            # d. plot
        
            if self.plott:
                
                print('\nPlotting...')
            
                ##### Policy Functions #####
                for iz in range(self.Nz):
                    plt.plot(self.grid_a, self.VF[iz,:])
                plt.title('Value Function')
                plt.legend(['z='+str(self.grid_z[0]),'z='+str(self.grid_z[1])])
                plt.xlabel('Assets')
                plt.savefig('value_function_vfi_aiyagari_small.pdf')
                plt.show()
                
                for iz in range(self.Nz):
                    plt.plot(self.grid_a, self.pol_sav[iz,:])
                plt.title("Savings Policy Function")
                plt.plot([self.a_bar,self.a_max], [self.a_bar,self.a_max],linestyle=':')
                plt.legend(['z='+str(self.grid_z[0]),'z='+str(self.grid_z[1]),'45 degree line'])
                plt.xlabel('Assets')
                plt.savefig('savings_policyfunction_vfi_aiyagari_small.pdf')
                plt.show()
                
                for iz in range(self.Nz):
                    plt.plot(self.grid_a, self.pol_cons[iz,:])
                plt.title("Consumption Policy Function")
                plt.legend(['z='+str(self.grid_z[0]),'z='+str(self.grid_z[1])])
                plt.xlabel('Assets')
                plt.savefig('consumption_policyfunction_vfi_aiyagari_small.pdf')
                plt.show()
                
                
                ##### Asset Distribution ####
                sns.histplot(self.ss_sim_a,  bins=100, stat='density')
                plt.xlabel('Assets')
                plt.title('Wealth Distribution')
                plt.savefig('wealth_distrib_vfi_aiyagari_small.pdf')
                plt.show()
                
            if self.plot_supply_demand:
                print('Plotting supply and demand...')
                
                self.ret_vec = np.linspace(-0.01,self.rho-0.001,8)
                self.k_demand, self.k_supply = self.graph_supply_demand(self.ret_vec)
                
                plt.plot(self.k_demand,self.ret_vec)
                plt.plot(self.k_supply,self.ret_vec)
                plt.plot(self.k_supply,np.ones(self.ret_vec.size)*self.rho,'--')
                plt.title('Capital Market')
                plt.legend(['Demand','Supply','Supply in CM'])
                plt.xlabel('Capital')
                plt.ylabel('Interest Rate')
                plt.savefig('capital_supply_demand.pdf')
                plt.show()
                
    
            t5 = time.time()
            print(f'Plot time elapsed: {t5-t4:.2f} seconds')
            
            print("\n-----------------------------------------")
            print("Stationary Equilibrium Solution")
            print("-----------------------------------------")
            print(f"Steady State Interest Rate = {ret_guess:.5f}")
            print(f"Steady State Capital = {self.k_ss:.2f}")
            print(f"Precautionary Savings Rate = {self.precaution_save:.5f}")
            print(f"Capital stock in incomplete markets is {((self.k_ss - self.k_cm)/self.k_cm)*100:.2f} percent higher than with complete markets")
            print('\nTotal iteration time elapsed: '+str(time.strftime("%M:%S",time.gmtime(t5-t0))))
            
            
                










#########################
# II. Jitted Functions  #
########################

################################
# 1. Helper Functions  #
###############################

@njit
def u(c, sigma):
    eps = 1e-8
    if  sigma == 1:
        return np.log(np.fmax(c, eps))
    else:
        return (np.fmax(c, eps) ** (1 - sigma) -1) / (1 - sigma)
    


###############################################
# 2. Household and Value Function Iteration  #
##############################################


@njit(parallel=True)
def solve_hh(ret, Nz, Na, tol, maxit, grid_a, w, grid_z, sigma, beta, pi):
   
    """
    Solves the household problem.
    
    *Output
        * VF is value function
        * pol_sav is the a' (savings) policy function
        * pol_cons is the consumption policy function
        * it_hh is the iteration number 
    """

    # a. Initialize counters, initial guess. storage matriecs
    dist = np.inf
    
    VF_old    = np.zeros((Nz,Na))  #initial guess
    VF = np.copy(VF_old)
    pol_sav = np.copy(VF_old)
    pol_cons = np.copy(VF_old)
    indk = np.copy(VF_old)
    
    # b. Iterate
    for it_hh in range(maxit) :
       for iz in range(Nz):
           for ia in prange(Na):
               c = (1+ret)*grid_a[ia] + w*grid_z[iz] - grid_a
               util = u(c, sigma)
               util[c < 0] = -10e9
               Tv = util + beta*(np.dot(pi[iz,:], VF_old))
               ind = np.argmax(Tv)
               VF[iz,ia] = Tv[ind]
               indk[iz,ia] = ind
               pol_sav[iz,ia] = grid_a[ind]
           
           # obtain consumption policy function
           pol_cons[iz,:] = (1+ret)*grid_a + w*grid_z[iz] - pol_sav[iz,:]
       
       dist = np.linalg.norm(VF-VF_old)
       
       if dist < tol :
           break
       
       VF_old = np.copy(VF)

    
    return VF, pol_sav, pol_cons, it_hh



####################
# 3. Simulation   #
##################

@njit(parallel=True)
def simulate_MonteCarlo( 
    a0,
    z0,
    sim_ret,
    sim_w,
    simN,
    simT,
    grid_z,
    grid_a,
    pol_cons,
    pol_sav,
    pi,
    seed,
        ):
    
    """
    Monte Carlo simulation for T periods for N households. Also checks 
    the grid size by ensuring that no more than 1% of households are at
    the maximum value of the grid.
    
    *Output
        - sim_k : aggregate capital (total savings in previous period)
        - sim_sav: current savings (a') profile
        - sim_z: income profile index, 0 for low state, 1 for high state
        - sim_c: consumption profile
        - sim_m: cash-on-hand profile ((1+r)a + w*z)
    """
    
    
    
    # 1. initialization
    np.random.seed(seed)
    sim_sav = np.zeros(simN)
    sim_c = np.zeros(simN)
    sim_m = np.zeros(simN)
    sim_z = np.zeros(simN, np.int32)
    sim_k = np.zeros(simT)
    edge = 0
    
    # 2. savings policy function interpolant
    polsav_interp = lambda a, z: interp(grid_a, pol_sav[z, :], a)
    
    # 3. simulate markov chain
    for t in range(simT):   #time

        draw = np.linspace(0, 1, simN)
        np.random.shuffle(draw)
        
        #calculate cross-sectional moments
        if t <= 0:
            sim_k[t] = np.mean(a0)
        else:
            sim_k[t] = np.mean(sim_sav)
        
        for i in prange(simN):  #individual

            # a. states 
            if t == 0:
                z_lag = np.int32(z0[i])
                a_lag = a0[i]
            else:
                z_lag = sim_z[i]
                a_lag = sim_sav[i]
                
            # b. shock realization. 0 for low state. 1 for high state.
            if draw[i] <= pi[z_lag, 1]:     #state transition condition
                sim_z[i] = 1
            else:
                sim_z[i] = 0
                
            # c. income
            y = sim_w[t]*grid_z[sim_z[i]]
            
            # d. cash-on-hand
            sim_m[i] = (1 + sim_ret[t]) * a_lag + y
            
            # e. consumption path
            sim_c[i] = sim_m[i] - polsav_interp(a_lag,sim_z[i])
            
            if sim_c[i] == pol_cons[sim_z[i],-1]:
                edge = edge + 1
            
            # f. savings path
            sim_sav[i] = sim_m[i] - sim_c[i]
            
    # 4. grid size evaluation
    frac_outside = edge/grid_a.size
    if frac_outside > 0.01 :
        print('\nIncrease grid size')

    return sim_k, sim_sav, sim_z, sim_c, sim_m



#run everything

ge_vfi_small = AiyagariVFISmall()
ge_vfi_small.solve_model()
