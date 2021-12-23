
"""
Author: Jacob Hess 
Date: January 2021

Written in python 3.8 on Spyder IDE.

Description: Finds the stationary equilibrium in a production economy with incomplete markets and idiosyncratic income
risk as in Aiyagari (1994). Features of the algorithm are:
    
    1) endogenous grid method to solve the household problem
    2) discrete approximation up to 7 states of a continuous AR(1) income process using the Rouwenhorst method 
    (Tauchen is also an option in the code)
    3) approximation of the stationary distribution using a monte carlo simulation
    
Aknowledgements: I used notes or pieces of code from the following :
    1) Gianluca Violante's notes (https://sites.google.com/a/nyu.edu/glviolante/teaching/quantmacro)
    2) Jeppe Druedahl (https://github.com/JeppeDruedahl) and NumEconCopenhagen (https://github.com/NumEconCopenhagen)
    
Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual)
    -- QuantEcon (to install: 'conda install quantecon')
    -- Interpolation from EconForge
       * optimized interpolation routines for python/numba
       * to install 'conda install -c conda-forge interpolation'
       * https://github.com/EconForge/interpolation.py

Note: If simulation tells you to increase grid size, increase self.sav_max in function setup_parameters.
"""


import time
import numpy as np
from numba import njit, prange
import quantecon as qe
from scipy.stats import rv_discrete
from interpolation import interp
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')




#############
# I. Model  #
############

class AiyagariEGM:
    
    """
    Class object of the model. AiyagariEGM().solve_model() runs everything
    """

    ############
    # 1. setup #
    ############

    def __init__(self, sigma=2,               #crra coefficient
                       rho_z = 0.9,           #autocorrelation coefficient
                       sigma_u = 0.2,         #std. dev. of shocks at annual frequency
                       Nz = 7,                #number of discrete income states
                       z_bar = 0,             #constant term in continuous income process (not the mean of the process)
                       a_bar = 0,             #select borrowing limit
                       plott =1,              #select 1 to make plots
                       plot_supply_demand = 1 # select 1 for capital market supply/demand graph
                       ):
        
        #parameters subject to changes
        self.sigma, self.a_bar, self.plott, self.plot_supply_demand  = sigma, a_bar, plott, plot_supply_demand
        self.rho_z, self.sigma_u, self.Nz, self.z_bar = rho_z, sigma_u, Nz, z_bar 
        
        self.setup_parameters()
        self.setup_grid()
        self.setup_discretization()
        

    def setup_parameters(self):

        # a. model parameters
        self.beta = 0.96  # discount factor
        self.rho = (1-self.beta)/self.beta #discount rate
        self.delta = 0.08  # depreciation rate
        self.alpha = 0.36  # cobb-douglas coeffient

        # b. hh solution
        self.tol_hh = 1e-6  # tolerance for consumption function iterations
        self.maxit_hh = 2000  # maximum number of iterations when finding consumption function in hh problem
       
        # savings grid
        self.Ns = 50
        self.sav_min = self.a_bar
        self.sav_max = 40
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
        # a. savings grid
        self.grid_sav = self.make_grid(self.sav_min, self.sav_max, self.Ns, self.curv) 

        
    def setup_discretization(self):
        
        # a. discretely approximate the continuous income process 
        self.mc = qe.markov.approximation.rouwenhorst(self.Nz, self.z_bar, self.sigma_u, self.rho_z)
        #self.mc = qe.markov.approximation.tauchen(self.rho_z, self.sigma_u, self.z_bar, 3, self.Nz)

        # b. transition matrix and states
        self.pi = self.mc.P
        self.grid_z = np.exp(self.mc.state_values)
        
        # c. initial distribution of z
        z_diag = np.diag(self.pi ** 1000)
        self.ini_p_z = z_diag / np.sum(z_diag)
        
        # d. idiosyncratic shock simulation for each household
        self.shock_matrix= np.zeros((self.ss_simT, self.ss_simN))
            
        # initial z shock drawn from initial distribution
        random_z = rv_discrete(values=(np.arange(self.Nz),self.ini_p_z),seed=self.seed)
        self.z0_idx = random_z.rvs(size=self.ss_simN)   #returns shock index, not grid value 
        
        # idiosyncratic income shock index realizations for all individuals
        for n in range(self.ss_simN) :
            self.shock_matrix[:,n] = self.mc.simulate_indices(self.ss_simT, init=self.z0_idx[n])
        
        
    
    
    
    
    
    
    
    
    
    
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
    
    
    


    
    
    ############################################
    # 3. Household and Endogenous Grid Method #
    ###########################################
     
    def solve_egm(self, pol_cons_old, ret_ss, w_ss):
           
        """
        Endogenous grid method to help solve the household problem
        """
           
        # a. initialize 
        c_tilde=np.empty((self.Nz,self.Ns))
        a_star=np.empty((self.Nz,self.Ns))
        pol_cons = np.empty((self.Nz,self.Ns))
        
        for i_z in range(self.Nz):
     
            # b. find RHS of euler equation (step 3 in EGM algo)
            avg_marg_u_plus = np.zeros(self.Ns)
            for i_zz in range(self.Nz):
     
                # i. future consumption
                c_plus = pol_cons_old[i_zz,:]
     
                # iii. future marginal utility
                marg_u_plus = self.u_prime(c_plus)
     
                # iv. average marginal utility
                weight = self.pi[i_z, i_zz]
     
                avg_marg_u_plus += weight * marg_u_plus
                
            ee_rhs = (1 + ret_ss) * self.beta * avg_marg_u_plus    
     
            # b. find current consumption (step 4 EGM algo)
            c_tilde[i_z,:] = self.u_prime_inv(ee_rhs)
            
            # c. get the endogenous grid of the value of assets today (step 5 EGM algo) 
            a_star[i_z,:] = (c_tilde[i_z,:] + self.grid_sav - self.grid_z[i_z]*w_ss) / (1+ret_ss)
            
            # d. update new consumption policy guess on savings grid
            for i_s, v_s in enumerate(self.grid_sav):
                if v_s <= a_star[i_z,0]:   #borrowing constrained, outside the grid range on the left
                    pol_cons[i_z, i_s] = (1+ret_ss)*v_s + self.grid_sav[0] + self.grid_z[i_z]*w_ss
                    
                elif  v_s >= a_star[i_z,-1]: # , linearly extrapolate, outside the grid range on the right
                    pol_cons[i_z, i_s] = c_tilde[i_z,-1] + (v_s-a_star[i_z,-1])*(c_tilde[i_z,-1] - c_tilde[i_z,-2])/(a_star[i_z,-1]-a_star[i_z,-2])
     
                else: #linearly interpolate, inside the grid range
                    pol_cons[i_z, i_s] = interp(a_star[i_z,:], c_tilde[i_z,:], v_s)
    
        
    
        return pol_cons, a_star
   
    def solve_hh(self, ret_ss, w_ss):
        
        """
        Solves the household problem.
        """
    
        # a. initial guess (consume everything. Step 2 in EGM algo)
        
        pol_cons_old = np.empty((self.Nz,self.Ns))
        for i_z, v_z in enumerate(self.grid_z):
            pol_cons_old[i_z,:] = (1+ret_ss)*self.grid_sav + v_z*w_ss

        # b. policy function iteration
        
        for self.it_hh in range(self.maxit_hh):
            
            # i. iterate
            pol_cons, a_star = self.solve_egm(pol_cons_old, ret_ss, w_ss)
            
            # ii. distance
            diff_hh = np.abs(pol_cons - pol_cons_old).max()
            #diff_hh = np.linalg.norm(c_hat_new - c_hat)
            
            if diff_hh < self.tol_hh :
                break
            
            pol_cons_old = np.copy(pol_cons)

        # c. obtain savings policy function
        pol_sav = np.empty([self.Nz, self.Ns])
        for i_z, v_z in enumerate(self.grid_z):
            pol_sav[i_z,:] = (1+ret_ss)*self.grid_sav + v_z*w_ss - pol_cons[i_z,:]
            
        
        return pol_cons, pol_sav, a_star

    
    
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
            pol_cons_graph, pol_sav_graph, a_star_graph = self.solve_hh(ret_graph, w_graph)
            
            
            # ii. Simulation
            a0 = self.ss_a0 * np.ones(self.ss_simN)
            z0 = self.grid_z[self.z0_idx]
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
                self.grid_sav,
                pol_cons_graph,
                pol_sav_graph,
                self.pi,
                self.shock_matrix,
            )
            
            k_supply[idx] = np.mean(sim_k_graph[self.ss_sim_burnin:])
            
        # 4. plot
            
        plt.plot(k_demand,ret_vec)
        plt.plot(k_supply,ret_vec)
        plt.plot(k_supply,np.ones(ret_vec.size)*self.rho,'--')
        plt.title('Capital Market')
        plt.legend(['Demand','Supply','Supply in CM'])
        plt.xlabel('Capital')
        plt.ylabel('Interest Rate')
        plt.savefig('capital_supply_demand.pdf')
        plt.show()

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
        
        self.pol_cons, self.pol_sav, self.a_star = self.solve_hh(self.ret_ss, self.w_ss)
        
        if self.it_hh < self.maxit_hh:
            print(f"Policy function convergence in {self.it_hh} iterations.")
        else : 
            raise Exception("No policy function convergence.")


            
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
            self.grid_sav,
            self.pol_cons,
            self.pol_sav,
            self.pi,
            self.shock_matrix
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
            z0 = self.grid_z[self.z0_idx]
    
            # b. initial interest rate guess (step 1)
            ret_guess = 0.03       
            
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
                    print("\n-----------------------------------------")
                    print('\nConvergence!')
                    break
                else :
                    #adaptive dampening 
                    if np.abs(diff) > np.abs(diff_old):
                        ret_guess = ret_guess - self.dp_small*diff  #to prevent divergence force a conservative new guess
                    else:
                        ret_guess = ret_guess - self.dp_big*diff
                    
                    print(f"\nNew interest rate guess = {ret_guess:.5f} \t diff = {diff:8.5f}")
                    diff_old=diff
            
            if it > self.maxit :
                print("No convergence")
                
            #stationary equilibrium prices and precautionary savings rate
            self.ret_ss = ret_guess
            self.w_ss = self.w_func(self.ret_ss)
            self.precaution_save = self.ret_cm - self.ret_ss
            
            t4 = time.time()
            print('Total iteration time elapsed: '+str(time.strftime("%M:%S",time.gmtime(t4-t0))))
            
            # d. plot
        
            if self.plott:
                
                print('\nPlotting...')
            
                ##### Policy Functions #####
                plt.plot(self.grid_sav, self.pol_sav.T)
                plt.title("Savings Policy Function")
                plt.xlabel('Assets')
                plt.savefig('savings_policyfunction_egm_aiyagari.pdf')
                plt.show()
                
                plt.plot(self.grid_sav, self.pol_cons.T)
                plt.title("Consumption Policy Function")
                plt.xlabel('Assets')
                plt.savefig('consumption_policyfunction_egm_aiyagari.pdf')
                plt.show()
                
                
                ##### Asset Distribution ####
                sns.histplot(self.ss_sim_a,  bins=100, stat='density')
                plt.xlabel('Assets')
                plt.title('Wealth Distribution')
                plt.savefig('wealth_distrib_egm_aiyagari.pdf')
                plt.show()
                
            if self.plot_supply_demand:
                print('\nPlotting supply and demand...')
                
                self.ret_vec = np.linspace(-0.01,self.rho-0.001,8)
                self.k_demand, self.k_supply = self.graph_supply_demand(self.ret_vec)
                
                
                
    
            t5 = time.time()
            print(f'Plot time elapsed: {t5-t4:.2f} seconds')
            
            print("\n-----------------------------------------")
            print("Stationary Equilibrium Solution")
            print("-----------------------------------------")
            print(f"Steady State Interest Rate = {ret_guess:.5f}")
            print(f"Steady State Capital = {self.k_ss:.2f}")
            print(f"Precautionary Savings Rate = {self.precaution_save:.5f}")
            print(f"Capital stock in incomplete markets is {((self.k_ss - self.k_cm)/self.k_cm)*100:.2f} percent higher than with complete markets")
            print('\nTotal run time: '+str(time.strftime("%M:%S",time.gmtime(t5-t0))))
            
            
                










#########################
# II. Jitted Functions  #
########################

####################
# 1. Simulation   #
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
    grid_sav,
    pol_cons,
    pol_sav,
    pi,
    shock_matrix
        ):
    
    """
    Monte Carlo simulation for T periods for N households. Also checks 
    the grid size by ensuring that no more than 1% of households are at
    the maximum value of the grid.
    
    *Output
        - sim_k : aggregate capital (total savings in previous period)
        - sim_sav: current savings (a') profile
        - sim_z: income profile 
        - sim_c: consumption profile
        - sim_m: cash-on-hand profile ((1+r)a + w*z)
    """
    
    
    
    # 1. initialization
    sim_sav = np.zeros(simN)
    sim_c = np.zeros(simN)
    sim_m = np.zeros(simN)
    sim_z = np.zeros(simN, np.float64)
    sim_z_idx = np.zeros(simN, np.int32)
    sim_k = np.zeros(simT)
    edge = 0
    
    # 2. savings policy function interpolant
    polsav_interp = lambda a, z: interp(grid_sav, pol_sav[z, :], a)
    
    # 3. simulate markov chain
    for t in range(simT):   #time
    
        #calculate cross-sectional moments
        if t <= 0:
            sim_k[t] = np.mean(a0)
        else:
            sim_k[t] = np.mean(sim_sav)
        
        for i in prange(simN):  #individual

            # a. states 
            if t == 0:
                a_lag = a0[i]
            else:
                a_lag = sim_sav[i]
                
            # b. shock realization 
            sim_z_idx[i] = shock_matrix[t,i]
            sim_z[i] = grid_z[sim_z_idx[i]]
                
            # c. income
            y = sim_w[t]*sim_z[i]
            
            # d. cash-on-hand
            sim_m[i] = (1 + sim_ret[t]) * a_lag + y
            
            # e. consumption path
            sim_c[i] = sim_m[i] - polsav_interp(a_lag,sim_z_idx[i])
            
            if sim_c[i] == pol_cons[sim_z_idx[i],-1]:
                edge = edge + 1
            
            # f. savings path
            sim_sav[i] = sim_m[i] - sim_c[i]
            
    # 4. grid size evaluation
    frac_outside = edge/grid_sav.size
    if frac_outside > 0.01 :
        print('\nIncrease grid size!')

    return sim_k, sim_sav, sim_z, sim_c, sim_m



#run everything

ge_egm = AiyagariEGM()
ge_egm.solve_model()
