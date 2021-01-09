
"""
Author: Jacob Hess 
Date: January 2021

Written in python 3.8 on Spyder IDE.

Description: This code solves the consumption/saving (aka the income flucuation problem) for the infinitely household 
in partial equilibrium using the endogenous grid method. The continuous income process is discretely approximated up to 
seven states using the Rouwenhorst method. It also runs a simulation and through that it calculates the 
euler equation error. 

Aknowledgements: I used notes or pieces of code from the following :
    1) Gianluca Violante's notes (https://sites.google.com/a/nyu.edu/glviolante/teaching/quantmacro)
    2) CompEcon Workshop (https://github.com/NYUEcon/CompEconWorkshop_2017)
    3) Jeppe Druedahl (https://github.com/JeppeDruedahl) and NumEconCopenhagen (https://github.com/NumEconCopenhagen)
    
Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual)
    -- Interpolation from EconForge
       * optimized interpolation routines for python/numba
       * to install 'conda install -c conda-forge interpolation'
       * https://github.com/EconForge/interpolation.py
    -- QuantEcon (to install: 'conda install quantecon')

Note: If simulation tells you to increase grid size, increase self.sav_max in function setup_parameters.
"""


import time
import numpy as np
from numba import njit, prange
from scipy.stats import rv_discrete
from interpolation import interp
import quantecon as qe
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')




#############
# I. Model  #
############

class ConSaveEGMV2:

    ############
    # 1. setup #
    ############

    def __init__(self, sigma=2,               #crra coefficient
                       rho_z = 0.9,          #autocorrelation coefficient
                       sigma_u = 0.2,         #std. dev. of shocks at annual frequency
                       Nz = 7,                 #number of discrete income states
                       z_bar = 0,             #constant term in continuous income process (not the mean of the process)
                       a_bar = 0,             #select borrowing limit
                       plott =1               #select 1 to make plots
                       ):
        
        #parameters subject to changes
        self.sigma, self.a_bar, self.plott =sigma, a_bar, plott
        self.rho_z, self.sigma_u, self.Nz, self.z_bar = rho_z, sigma_u, Nz, z_bar 

        self.setup_parameters()
        self.setup_grid()
        self.setup_discretization()
        
        # We need (1+r)beta < 1 for convergence.
        assert (1 + self.ret) * self.beta < 1, "Stability condition violated."
        
        #We require the borrowing limit to be greater than the natural borriwing limit (or no ponzi condition).
        #The limit is where an agent can borrow and repay it in the next period with probability 1.
        assert self.a_bar + 1e-6 > ((-1) * ((1+self.ret)/self.ret) * self.grid_z[0]), "Natural borrowing limit violated."
        

    def setup_parameters(self):

        # a. model parameters
        self.beta = 0.95  # discount factor
        self.rho = (1-self.beta)/self.beta #discount rate
    
        # prices 
        self.w=1 
        self.ret=0.04

        # b. hh solution
        self.tol_hh = 1e-6  # tolerance for policy functions iterations
        self.maxit_hh = 2000  # maximum number of policy functions iterations

        # savings grid
        self.Ns = 50
        self.sav_min = self.a_bar
        self.sav_max = 80
        self.curv = 3
        
        # c. simulation
        self.seed = 123
        self.a0 = 1.0  # initial assets
        self.simN = 50_000  # number of households
        self.simT =  2000 # number of time periods to simulate



    def setup_grid(self):

        # a. savings (or end-of-period assets) grid
        self.grid_sav = self.make_grid(self.sav_min, self.sav_max, self.Ns, self.curv)  

        
    def setup_discretization(self):
        
        # a. discretely approximate the continuous income process 
        self.mc = qe.markov.approximation.rouwenhorst(self.Nz, self.z_bar, self.sigma_u, self.rho_z)
        #self.mc = qe.markov.approximation.tauchen(self.rho_z, self.sigma_u, self.z_bar, 3, self.Nz)

        # b. transition matrix and states
        self.pi = self.mc.P
        self.pi_stat = self.mc.stationary_distributions.T
        self.grid_z = np.exp(self.mc.state_values)

        # c. initial distribution of z
        z_diag = np.diag(self.pi ** 1000)
        self.ini_p_z = z_diag / np.sum(z_diag)

        
        
        

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
    
    def interpol(self,x,y,x1):
        
        """
        1-D linear interpolation.
        
        *Input
            - x : x-coordinates to evaluate on
            - y : y-coordinates of data points
            - x1 : x-coordinates of points to interpolate
        *Output
            - y1 : interpolated value y-coordinate
            - i : grid index on the right of x1.
        """
        
        N = len(x)
        i = np.minimum(np.maximum(np.searchsorted(x,x1,side='right'),1),N-1)
        xl = x[i-1]
        xr = x[i]
        yl = y[i-1]
        yr = y[i]
        y1 = yl + (yr-yl)/(xr-xl) * (x1-xl)
        above = x1 > x[-1]
        below = x1 < x[0]
        y1 = np.where(above,y[-1] +   (x1 - x[-1]) * (y[-1]-y[-2])/(x[-1]-x[-2]), y1)
        y1 = np.where(below,y[0],y1)
    
        return y1, i
    
    
    
    
    

    ############################################
    # 3. Household and Endogenous Grid Method #
    ###########################################
         
    def solve_egm(self, pol_cons_old, ret, w):
           
           """
           Endogenous grid method to help solve the household problem
           """
           
           # a. initialize 
           c_tilde=np.empty((self.Nz,self.Ns))
           self.a_star=np.empty((self.Nz,self.Ns))
           self.pol_cons = np.empty((self.Nz,self.Ns))
           
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
                   
               ee_rhs = (1 + ret) * self.beta * avg_marg_u_plus    
    
               # b. find current consumption (step 4 EGM algo)
               c_tilde[i_z,:] = self.u_prime_inv(ee_rhs)
               
               # c. get the endogenous grid of the value of assets today (step 5 EGM algo) 
               self.a_star[i_z,:] = (c_tilde[i_z,:] + self.grid_sav - self.grid_z[i_z]*w) / (1+ret)
               
               # d. update new consumption policy guess on savings grid
               for i_s, v_s in enumerate(self.grid_sav):
                   if v_s <= self.a_star[i_z,0]:   #borrowing constrained, outside the grid range on the left
                       self.pol_cons[i_z, i_s] = (1+ret)*v_s + self.grid_sav[0] + self.grid_z[i_z]*w
                       
                   elif  v_s >= self.a_star[i_z,-1]: # , linearly extrapolate, outside the grid range on the right
                      self.pol_cons[i_z, i_s] = c_tilde[i_z,-1] + (v_s-self.a_star[i_z,-1])*(c_tilde[i_z,-1] - c_tilde[i_z,-2])/(self.a_star[i_z,-1]-self.a_star[i_z,-2])
    
                   else: #linearly interpolate, inside the grid range
                       self.pol_cons[i_z, i_s] = self.interpol(self.a_star[i_z,:], c_tilde[i_z,:], v_s)[0]
    

   
    def solve_hh(self):
        
        """
        Solves the household problem.
        """
    
        # a. initial guess (consume everything. Step 2 in EGM algo)
        
        self.pol_cons_old = np.empty((self.Nz,self.Ns))
        for i_z, v_z in enumerate(self.grid_z):
            self.pol_cons_old[i_z,:] = (1+self.ret)*self.grid_sav + v_z*self.w 

        # b. policy function iteration
        
        for self.it_hh in range(self.maxit_hh):
            
            # i. iterate
            self.solve_egm(self.pol_cons_old, self.ret, self.w)
            
            # ii. distance
            diff_hh = np.abs(self.pol_cons - self.pol_cons_old).max()
            #diff_hh = np.linalg.norm(c_hat_new - c_hat)
            
            if diff_hh < self.tol_hh :
                break
            
            self.pol_cons_old = np.copy(self.pol_cons)

        # c. obtain savings policy function
        self.pol_sav = np.empty([self.Nz, self.Ns])
        for i_z, v_z in enumerate(self.grid_z):
            self.pol_sav[i_z,:] = (1+self.ret)*self.grid_sav + v_z*self.w - self.pol_cons[i_z,:]
            
        
        
    
    
    
    
    #####################
    # 4. Main Function  #
    #####################


    def solve_model(self):
    
        """
        Runs the entire model.
        """    
        
        t0 = time.time()    #start the clock
        
        # a. solve household problem 
        print("\nSolving household problem...")
        
        self.solve_hh()
        
        if self.it_hh < self.maxit_hh:
            print(f"Policy function convergence in {self.it_hh} iterations.")
        else : 
            raise Exception("No policy function convergence.")
        
        
        t1 = time.time()
        print(f'Household problem time elapsed: {t1-t0:.2f} seconds')
        
        
        
        # b. simulation
        print("\nSimulating...")
        
        # i. initial values for agents
        a0 = self.a0 * np.ones(self.simN)
        self.shock_matrix= np.zeros((self.simT, self.simN))
        
        # initial z shock drawn from initial distribution
        random_z = rv_discrete(values=(np.arange(self.Nz),self.ini_p_z),seed=self.seed)
        z0_idx = random_z.rvs(size=self.simN)
        z0 = self.grid_z[z0_idx]
        
        
        for n in range(self.simN) :
            self.shock_matrix[:,n] = self.mc.simulate_indices(self.simT, init=z0_idx[n])
        
        
        # ii. simulate markov chain and endog. variables
        self.sim_c, self.sim_sav, self.sim_z, self.sim_m, self.euler_error = simulate_MarkovChain(
            a0,
            z0,
            self.ret,
            self.w,
            self.simN,
            self.simT,
            self.grid_z,
            self.grid_sav,
            self.pol_cons,
            self.pol_sav,
            self.a_bar,
            self.sigma,
            self.beta,
            self.pi,
            self.shock_matrix,
        )
        
        t2 = time.time()
        print(f'Simulation time elapsed: {t2-t1:.2f} seconds')
        
        
        # c. plot
        
        if self.plott:
            
            print('\nPlotting...')
            
            ##### Policy Functions #####
            plt.plot(self.grid_sav, self.pol_sav.T)   
            plt.title("Savings Policy Function")
            plt.xlabel('Assets')
            plt.savefig('savings_policyfunction_egmv2.pdf')
            plt.show()
            
            plt.plot(self.grid_sav, self.pol_cons.T)
            plt.title("Consumption Policy Function")
            plt.xlabel('Assets')
            plt.savefig('consumption_policyfunction_egmv2.pdf')
            plt.show()
            
            ##### Simulation ####
            fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,6))
            fig.tight_layout(pad=4)
            
            #first individual over first 100 periods
            ax1.plot(np.arange(0,99,1), self.sim_sav[:99,1], np.arange(0,99,1), self.sim_c[:99,1],
                     np.arange(0,99,1), self.sim_z[:99,1],'--')
            ax1.legend(['Savings', 'Consumption', 'Income'])  
            ax1.set_title('Simulation of First Household During First 100 Periods')
            
            #averages over entire simulation
            ax2.plot(np.arange(0,self.simT,1), np.mean(self.sim_sav, axis=1), 
                     np.arange(0,self.simT,1), np.mean(self.sim_c, axis=1) )
            ax2.legend(['Savings', 'Consumption', 'Income'])
            ax2.set_title('Simulation Average over 50,000 Households')
            plt.savefig('simulation_egmv2.pdf')
            plt.show()
            
            ##### Wealth Density #####
            sns.histplot(self.sim_sav[-1,:], bins=100, stat='density')
            plt.title("Marginal Wealth Density")
            plt.xlabel('Assets')
            plt.savefig('wealth_density_egmv2.pdf')
            plt.show()
            
            t3 = time.time()
            print(f'Plot time elapsed: {t3-t2:.2f} seconds')

        


        # d. euler error evaluation
    
        #maximum error
        self.ee_max = np.nanmax(self.euler_error)
        
        #average error for each individual 
        avg_individ = np.nanmean(self.euler_error,axis=1)
        
        #across all individuals
        self.ee_avg = np.nanmean(avg_individ)
        
        print("\n-----------------------------------------")
        print("Euler Equation Error Evaluation")
        print("-----------------------------------------")
        print(f"max error  = {self.ee_max:.5f}")
        print(f"average error = {self.ee_avg:.5f}")
        
        
        t4 = time.time()
        print(f'\nTotal Run Time: {t4-t0:.2f} seconds')




#########################
# II. Jitted Functions #
########################

####################
# 1. Simulation   #
##################

@njit(parallel=True)
def simulate_MarkovChain( 
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
    a_bar,
    sigma,
    beta,
    pi,
    shock_matrix
        ):
    
    """
    Simulates markov chain for T periods for N households. Also checks 
    the grid size by ensuring that no more than 1% of households are at
    the maximum value of the grid.
    
    *Output
        - sim_c: consumption profile
        - sim_sav: savings (a') profile
        - sim_z: income shock profile
        - sim_m: cash-on-hand profile ((1+r)a + w*z)
    """
    
    
    # 1. initialization
    
    sim_sav = np.zeros((simT,simN))
    sim_c = np.zeros((simT,simN))
    sim_m = np.zeros((simT,simN))
    sim_z = np.zeros((simT,simN), np.float64)
    sim_z_idx = np.zeros((simT,simN), np.int32)
    edge = 0
    euler_error = np.empty((simT,simN)) * np.nan
    
    # 2. savings policy function interpolant
    
    polsav_interp = lambda a, z: interp(grid_sav, pol_sav[z, :], a)
    
    u_prime = lambda c : c**(-sigma)
    
    u_prime_inv = lambda x : x ** (-1/sigma)
    
    # 3. simulate markov chain
    for t in range(simT):   #time
        
        for i in prange(simN):  #individual hh

            # a. states 
            if t == 0:
                a_lag = a0[i]
            else:
                a_lag = sim_sav[t-1,i]
                
            # b. shock realization. 
            sim_z_idx[t,i] = shock_matrix[t,i]
            sim_z[t,i] = grid_z[sim_z_idx[t,i]]
                
            # c. income
            y = sim_w*sim_z[t,i]
            
            # d. cash-on-hand path
            sim_m[t, i] = (1 + sim_ret) * a_lag + y
            
            # e. consumption path
            sim_c[t,i] = sim_m[t, i] - polsav_interp(a_lag,sim_z_idx[t,i])
            
            # f. savings path
            sim_sav[t,i] = sim_m[t, i] - sim_c[t,i]
            
           
                
            # g. error evaluation
            
            check_out=False
            if sim_sav[t,i] == pol_sav[sim_z_idx[t,i],-1]:
                edge = edge + 1
                check_out=True
                
            constrained=False
            if sim_sav[t,i] == 0:
                constrained=True
            
                
            if sim_c[t,i] < sim_m[t,i] and constrained==False and check_out==False :
                
                #all possible consumption choices tomorrow (2 in total)
                c_plus = np.empty(len(grid_z))
                
                for iz in range(len(grid_z)):
                    c_plus[iz] = (1 + sim_ret) * sim_sav[t,i] + sim_w*grid_z[iz] - polsav_interp(sim_sav[t,i],iz)
                
                #expectation of marginal utility of consumption
                avg_marg_c_plus = np.dot(pi[sim_z_idx[t,i],:], u_prime(c_plus))
                
                #euler error
                euler_error[t,i] = np.abs(1 - (u_prime_inv(beta*(1+sim_ret)*avg_marg_c_plus) / sim_c[t,i]))
            
            
            
            
            
    # 4. grid size evaluation
    frac_outside = edge/grid_sav.size
    if frac_outside > 0.01 :
        raise Exception('Increase grid size!')
    
    
                

    return sim_c, sim_sav, sim_z, sim_m, euler_error



#run everything

test=ConSaveEGMV2()
test.solve_model()





