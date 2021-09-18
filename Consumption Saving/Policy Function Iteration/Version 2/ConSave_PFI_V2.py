
"""
Author: Jacob Hess 
First Version: September 2021

Description: This code solves the consumption/saving problem (aka the income flucuation problem) for the infinitely 
household in partial equilibrium using policy function iteration on the euler equation with linear interpolation. 
The continuous income process is discretely approximated using the Rouwenhorst method.

To find the stationary distribution one can choose from three methods: 

1) Discrete approximation of the density function which conducts a fixed point iteration with linear interpolation
2) Eigenvector method to solve for the exact stationary density.
3) Monte carlo simulation. 
    
Finally, to evaluate the accuracy of the solution the code computes the euler equation error with two different methods. 
One by simulating the model and calulating the error for each individual. The other is by calculating the error across 
the entire in the state space.

Aknowledgements: I wrote the algorithms using the following resources :
    1) Gianluca Violante's global methods and distribution approximation notes (https://sites.google.com/a/nyu.edu/glviolante/teaching/quantmacro)
    2) Heer and Maussner 2nd ed. Ch. 7
    3) Raul Santaeulalia-Llopis' ABHI Notes (http://r-santaeulalia.net/)
    4) Alexander Ludwig's notes (https://alexander-ludwig.com/)
    5) Jeppe Druedahl (https://github.com/JeppeDruedahl) and NumEconCopenhagen (https://github.com/NumEconCopenhagen)
    
Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual)
    -- QuantEcon (to install: 'conda install quantecon')
    -- Interpolation from EconForge
       * optimized interpolation routines for python/numba
       * to install 'conda install -c conda-forge interpolation'
       * https://github.com/EconForge/interpolation.py

Note 1: If simulation tells you to increase grid size, increase self.a_max in function setup_parameters.
"""

import time
import numpy as np
from numba import njit, prange
from scipy.stats import rv_discrete
from interpolation import interp
import quantecon as qe
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')


#############
# I. Model  #
############

class ConSavePFI:    

    ############
    # 1. Setup #
    ############

    def __init__(self, a_bar = 0,              #select borrowing limit
                      plott =1,               #select 1 to make plots
                      simulate =0,            #select 1 to run simulation (if distribution_method = 'monte carlo' simulate is automatically set to 1 )
                      full_euler_error = 0,        #select 1 to compute euler_error for entire state space
                      distribution_method = 'discrete' #Approximation method of the stationary distribution. 
                                                      #Options: 'discrete', 'eigenvector', 'monte carlo' or 'none'
                      ):
        
        
        #parameters subject to changes
        self.a_bar, self.plott, self.simulate = a_bar, plott, simulate
        self.distribution_method, self.full_euler_error = distribution_method, full_euler_error

        self.setup_parameters()
        self.setup_grid()
        self.setup_discretization()
        
        #pack parameters for jitted functions
        
        self.params_pfi = self.r, self.w, self.beta, self.pi, self.grid_a, self.grid_z, self.sigma, self.maxit, self.tol
        
        if distribution_method == 'discrete':
            self.params_discrete = self.grid_a, self.grid_a_fine, self.Nz, self.pi, self.pi_stat, self.maxit, self.tol
            
        if self.simulate ==1 or self.distribution_method == 'monte carlo':
            self.params_sim = self.a0, self.z0, self.r, self.w, self.simN, self.simT, self.grid_z, self.grid_a, \
                self.sigma, self.beta, self.pi, self.shock_history
        
        # warnings
        
        # We need (1+r)beta < 1 for convergence.
        assert (1 + self.r) * self.beta < 1, "Stability condition violated."
        
        #We require the borrowing limit to be greater than the natural borriwing limit (or no ponzi condition).
        #The limit is where an agent can borrow and repay it in the next period with probability 1.
        assert self.a_bar + 1e-6 > ((-1) * ((1+self.r)/self.r) * self.grid_z[0]), "Natural borrowing limit violated."
        
        if self.distribution_method != 'discrete' and self.distribution_method != 'eigenvector' and self.distribution_method != 'monte carlo' and self.distribution_method != 'none' :
            raise Exception("Stationary distribution approximation method incorrectly entered: Choose 'discrete', 'eigenvector', 'monte carlo' or 'none' ")
            
        if self.plott != 1 and self.plott != 0:
            raise Exception("Plot option incorrectly entered: Choose either 1 or 0.")
            
        if self.simulate != 1 and self.simulate != 0:
            raise Exception("Simulate option incorrectly entered: Choose either 1 or 0.")
            
        if self.full_euler_error != 1 and self.full_euler_error != 0:
            raise Exception("Euler error full grid evaluation option incorrectly entered: Choose either 1 or 0.")
            
            
            
    def setup_parameters(self):

        # a. model parameters
        self.sigma = 2               #crra coefficient
        self.beta = 0.95  # discount factor
        self.rho = (1-self.beta)/self.beta #discount rate
    
        # AR(1) income process
        self.Nz = 7                 #number of discrete income states
        self.z_bar = 0             #constant term in continuous income process (not the mean of the process)
        self.rho_z = 0.9    #autocorrelation coefficient
        self.sigma_z = 0.2         #std. dev. of income process at annual frequency
    
        # prices 
        self.w=1 
        self.r=0.04

        # b. iteration parameters
        self.tol = 1e-6  # tolerance for iterations
        self.maxit = 2000  # maximum number of vf iterations
        
        
        # c. hh solution

        # asset grid 
        self.Na = 200
        self.a_min = self.a_bar
        self.a_max = 60
        self.curv = 3 
        
        if self.distribution_method == 'discrete' or self.distribution_method == 'eigenvector' or self.full_euler_error:
            self.Na_fine = self.Na*3
        
        # d. simulation
        if self.simulate or self.distribution_method == 'monte carlo':
            self.seed = 123
            self.simN = 50_000  # number of households
            self.simT =  2000 # number of time periods to simulate
            self.sim_burnin = 1000  # burn-in periods before calculating average savings
            self.init_asset = 1.0  # initial asset (homogenous)
            self.a0 = self.init_asset * np.ones(self.simN)  #initial asset for all individuals
            
            

    def setup_grid(self):

        # a. asset grid
        self.grid_a = self.make_grid(self.a_min, self.a_max, self.Na, self.curv)  #asset grid
        
        # b. finer grid for density approximation and euler error
        if self.distribution_method == 'discrete' or self.distribution_method == 'eigenvector' or self.full_euler_error :
            self.grid_a_fine = self.make_grid(self.a_min, self.a_max, self.Na_fine, self.curv)   



    def setup_discretization(self):
        
        # a. discretely approximate the continuous income process 
        self.mc = qe.markov.approximation.rouwenhorst(self.Nz, self.z_bar, self.sigma_z, self.rho_z)
        #self.mc = qe.markov.approximation.tauchen(self.rho_z, self.sigma_z, self.z_bar, 3, self.Nz)

        # b. transition matrix and states
        self.pi = self.mc.P
        self.pi_stat = self.mc.stationary_distributions.T
        self.grid_z = np.exp(self.mc.state_values)

        # c. initial distribution of z
        z_diag = np.diag(self.pi ** 1000)
        self.ini_p_z = z_diag / np.sum(z_diag)
        
        # d. income shock sequence for each individual for simulation
        if self.simulate or self.distribution_method == 'monte carlo':
         
            # initial income shock drawn for each individual from initial distribution
            random_z = rv_discrete(values=(np.arange(self.Nz),self.ini_p_z), seed=self.seed)
            z0_idx = random_z.rvs(size=self.simN)
            self.z0 = self.grid_z[z0_idx]   #initial state for each individual
            
            # draw income shocks for each individual
            self.shock_history= np.zeros((self.simT, self.simN))
            
            for n in range(self.simN) :
                seed_sim = self.seed + n    #unique seed for each individual
                self.shock_history[:,n] = self.mc.simulate_indices(self.simT, init=z0_idx[n], random_state=seed_sim)       
        




    #######################
    # 2. helper functions #
    ######################
    
    def make_grid(self, min_val, max_val, num, curv):  
        """
        Makes an exponential grid of degree curv. A higher curv will put more points closer a_min. 
        
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
        
    
    
        
        
    ######################################
    # 3. Euler Equation Error Analysis  #
    #####################################
    

    def ee_error(self):
        """
        Computes the euler equation error over the entire state space with a finer grid.
        
        *Output
            * Log10 euler_error
            * max Log10 euler error
            * average Log10 euler error
        """
        
                
        # a. initialize
        euler_error = np.zeros((self.Nz, self.Na_fine))
        
        # b. helper function
        u_prime = lambda c : c**(-self.sigma)
        
        u_prime_inv = lambda x : x ** (-1/self.sigma)
        
        # c. calculate euler error at all fine grid points
        
        for i_z, z in enumerate(self.grid_z):       #current income shock
            for i_a, a in enumerate(self.grid_a_fine):   #current asset level
                
                # i. interpolate savings policy function fine grid point
            
                a_plus = interp(self.grid_a, self.pol_sav[i_z,:], a)
                
                # liquidity constrained, do not calculate error
                if a_plus <= 0:     
                    euler_error[i_z, i_a] = np.nan
                
                # interior solution
                else:
                    
                    # ii. current consumption and initialize expected marginal utility
                    c = (1 + self.r) * a + self.w * z - a_plus
                    avg_marg_c_plus = 0
                    
                    # iii. expected marginal utility
                    for i_zz, z_plus in enumerate(self.grid_z):      #next period productivity
                    
                        c_plus = (1 + self.r) * a_plus + self.w * z_plus - interp(self.grid_a, self.pol_sav[i_zz,:], a_plus)
                        
                        #expectation of marginal utility of consumption
                        avg_marg_c_plus += self.pi[i_z,i_zz] * u_prime(c_plus)
                    
                    # iv. compute euler error
                    euler_error[i_z, i_a] = 1 - u_prime_inv(self.beta*(1+self.r)*avg_marg_c_plus) / c
                    
       
        # ii. transform euler error with log_10. take max and average
        euler_error = np.log10(np.abs(euler_error))
        max_error =  np.nanmax(np.nanmax(euler_error, axis=1))
        avg_error = np.nanmean(euler_error) 
        
        
        
        return euler_error, max_error, avg_error
        
    
    
    
    
    ####################################################
    # 4. Stationary Distribution: Eigenvector Method   #
    ####################################################
    
    
    def eigen_stationary_density(self):
        """
        Solve for the exact stationary density. First constructs the Nz*Ns by Nz*Ns transition matrix Q(a',z'; a,z) 
        from state (a,z) to (a',z'). Then obtains the eigenvector associated with the unique eigenvalue equal to 1. 
        This eigenvector (renormalized so that it sums to one) is the unique stationary density function.
        
        Note: About 99% of the computation time is spend on the eigenvalue calculation. For now there is no
        way to speed this function up as numba only supports np.linalg.eig() when there is no domain change 
        (ex. real numbers to real numbers). Here there is a domain change as some eigenvalues and eigenvector 
        elements are complex.

        *Output
            * stationary_pdf: stationary density function
            * Q: transition matrix
        """
        
        # a. initialize transition matrix
        Q = np.zeros((self.Nz*self.Na_fine, self.Nz*self.Na_fine))
        
        # b. interpolate and construct transition matrix 
        for i_z in range(self.Nz):    #current productivity 
            for i_a, a0 in enumerate(self.grid_a_fine):    
                
                # i. interpolate
                a_intp = interp(self.grid_a, self.pol_sav[i_z,:], a0)
                
                #take the grid index to the right. a_intp lies between grid_sav_fine[j-1] and grid_sav_fine[j]. 
                j = np.sum(self.grid_a_fine <= a_intp) 
                
                    
                #less than or equal to lowest grid value
                if a_intp <= self.grid_a_fine[0]:
                    p = 0
                    
                #more than or equal to greatest grid value
                elif a_intp >= self.grid_a_fine[-1]:
                   p = 1
                   j = j-1 #since right index is outside the grid make it the max index
                   
                #inside grid
                else:
                   p = (a_intp-self.grid_a_fine[j-1]) / (self.grid_a_fine[j]-self.grid_a_fine[j-1])
                    
                # ii. transition matrix
                na = i_z*self.Na_fine    #minimum row index
                
                for i_zz in range(self.Nz):     #next productivity state
                    ma = i_zz * self.Na_fine     #minimum column index
                    
                    Q[na + i_a, ma + j]= p * self.pi[i_z, i_zz]
                    Q[na + i_a, ma + j - 1] = (1.0-p)*self.pi[i_z, i_zz]
        
        # iii. ensure that the rows sum up to 1
        assert np.allclose(Q.sum(axis=1), np.ones(self.Nz*self.Na_fine)), "Transition matrix error: Rows do not sum to 1"
        
        
        
        # c. get the eigenvector 
        eigen_val, eigen_vec = np.linalg.eig(Q.T)    #transpose Q for eig function.
        
        # i. find column index for eigen value equal to 1
        idx = np.argmin(np.abs(eigen_val-1.0))
        
        eigen_vec_stat = np.copy(eigen_vec[:,idx])
        
        
        
        # ii. ensure complex arguments of any complex numbers are small and convert to real numbers
        
        if np.max(np.abs(np.imag(eigen_vec_stat))) < 1e-6:
            eigen_vec_stat = np.real(eigen_vec_stat)  # drop the complex argument of any complex numbers. 
            
        else:
            raise Exception("Stationary eigenvector error: Maximum complex argument greater than 0.000001. Use a different distribution solution method.")
        
        
        # d. obtain stationary density from stationary eigenvector
        
        # i. reshape
        stationary_pdf = eigen_vec_stat.reshape(self.Nz,self.Na_fine)
        
        # ii. stationary distribution by percent 
        stationary_pdf=stationary_pdf/np.sum(np.sum(stationary_pdf,axis=0)) 
        
        return stationary_pdf, Q
    
    
    
    
    
    ######################
    # 5. Main Function  #
    #####################
    
    def solve_model(self):

        """
        Runs the entire model.
        """    
        
        t0 = time.time()    #start the clock
        
        
        
        # a. solve household problem
        print("\nSolving household problem...")
        
        self.pol_sav, self.pol_cons, self.it_hh = solve_hh(self.params_pfi)
        
        if self.it_hh < self.maxit-1:
            print(f"Policy function convergence in {self.it_hh} iterations.")
        else : 
            raise Exception("No policy function convergence.")
        
        t1 = time.time()
        print(f'Household problem time elapsed: {t1-t0:.2f} seconds')
            
        
        
        # b. stationary distribution
        
        # discrete approximation
        if self.distribution_method == 'discrete':
            
            print("\nStationary Distribution Solution Method: Discrete Approximation and Forward Iteration on Density Function")
            print("\nComputing...")
            
            # i. approximate stationary density
            self.stationary_pdf, self.it_pdf = discrete_stationary_density(self.pol_sav, self.params_discrete)
            
            if self.it_pdf < self.maxit-1:
                print(f"Convergence in {self.it_pdf} iterations.")
            else : 
                raise Exception("No density function convergence.")
            
            # ii. steady state assets
            self.a_ss = np.sum(np.dot(self.stationary_pdf, self.grid_a_fine))
            
            # iii. marginal wealth density
            self.stationary_wealth_pdf = np.sum(self.stationary_pdf, axis=0)
            
            t2 = time.time()
            print(f'Density approximation time elapsed: {t2-t1:.2f} seconds')
            
            
            
        # eigenvector
        if self.distribution_method == 'eigenvector':
            
            print("\nStationary Distribution Solution Method: Eigenvector Method for Exact Stationary Density")
            print("\nComputing...")
            
            self.stationary_pdf, self.Q = self.eigen_stationary_density()
            
            # i. aggregate asset holdings
            self.a_ss = np.sum(np.dot(self.stationary_pdf, self.grid_a_fine))
            
            # iii. marginal wealth density
            self.stationary_wealth_pdf = np.sum(self.stationary_pdf, axis=0)
            
            t2 = time.time()
            print(f'Density computation time elapsed: {t2-t1:.2f} seconds')
            
            
            
        # monte carlo simulation
        if self.simulate ==1 or self.distribution_method == 'monte carlo':
            
            if self.distribution_method == 'monte carlo':
                print("\nStationary Distribution Approximation: Monte Carlo Simulation")
            
            print("\nSimulating...")
            
            # i. simulate markov chain and endog. variables
            self.sim_c, self.sim_sav, self.sim_z, self.sim_m, self.euler_error_sim = simulate_MarkovChain(
                self.pol_cons,
                self.pol_sav,
                self.params_sim
            )
            
            # ii. steady state assets
            if self.distribution_method == 'monte carlo':
                self.a_ss = np.mean(self.sim_sav[self.sim_burnin :])
            
            # iii. max and average euler error error, ignores nan which is when the euler equation does not bind
            self.max_error_sim =  np.nanmax(self.euler_error_sim)
            self.avg_error_sim = np.nanmean(np.nanmean(self.euler_error_sim, axis=1)) 
            
            t2 = time.time()
            print(f'Simulation time elapsed: {t2-t1:.2f} seconds')
        
        
        
        else:
            t2 = time.time()
            
        
        
        # c. calculate euler equation error across the state space
        
        if self.full_euler_error:
            print("\nCalculating Euler Equation Error...")
            
            self.euler_error, self.max_error, self.avg_error = self.ee_error()
            
            t3 = time.time()
            print(f'Euler Eq. error calculation time elapsed: {t3-t2:.2f} seconds')
            
        else: 
            t3 = time.time()
        
        
        
        # d. plot
        
        if self.plott:
            
            print('\nPlotting...')
        
        
        
            ##### Solutions #####
            plt.plot(self.grid_a, self.pol_sav.T)
            plt.title("Savings Policy Function")
            plt.plot([self.a_bar,self.a_max], [self.a_bar,self.a_max],linestyle=':')
            plt.xlabel('Assets')
            #plt.savefig('savings_policyfunction_pfi_v2.pdf')
            plt.show()
            
            plt.plot(self.grid_a, self.pol_cons.T)
            plt.title("Consumption Policy Function")
            plt.xlabel('Assets')
            #plt.savefig('consumption_policyfunction_pfi_v2.pdf')
            plt.show()
            
            if self.full_euler_error:
                plt.plot(self.grid_a_fine, self.euler_error.T)
                plt.title('Log10 Euler Equation Error')
                plt.xlabel('Assets')
                #plt.savefig('log10_euler_error_pfi_v2.pdf')
                plt.show()
           
            
           
            ##### Distributions ####
            if self.distribution_method == 'discrete' or self.distribution_method == 'eigenvector':
                
                # marginal wealth density
                plt.plot(self.grid_a_fine, self.stationary_wealth_pdf)
                plt.title("Stationary Wealth Density (Discrete Approx.)") if self.distribution_method == 'discrete' else plt.title("Stationary Wealth Density (Eigenvector Method)")
                plt.xlabel('Assets')
                #plt.savefig('wealth_density_pfi_v2_discrete.pdf') if self.distribution_method == 'discrete' else plt.savefig('wealth_density_pfi_v2_eigenvector.pdf')
                plt.show()
                
            
            if self.distribution_method == 'monte carlo':
                sns.histplot(self.sim_sav[-1,:], bins=100, stat='density')
                plt.title("Stationary Wealth Density (Monte Carlo Approx.)")
                plt.xlabel('Assets')
                #plt.savefig('wealth_density_pfi_v2_montecarlo.pdf')
                plt.show()
           
            
           
            ##### Simulation #####
            if self.simulate or self.distribution_method == 'monte carlo':
                fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,6))
                fig.tight_layout(pad=4)
                
                #first individual over first 100 periods
                ax1.plot(np.arange(0,99,1), self.sim_sav[:99,0], np.arange(0,99,1), self.sim_c[:99,0],
                         np.arange(0,99,1), self.sim_z[:99,0],'--')
                ax1.legend(['Savings', 'Consumption', 'Income'])  
                ax1.set_title('Simulation of First Household During First 100 Periods')
                
                #averages over entire simulation
                ax2.plot(np.arange(0,self.simT,1), np.mean(self.sim_sav, axis=1), 
                         np.arange(0,self.simT,1), np.mean(self.sim_c, axis=1) )
                ax2.legend(['Savings', 'Consumption', 'Income'])
                ax2.set_title('Simulation Average over 50,000 Households')
                #plt.savefig('simulation_pfi_v2.pdf')
                plt.show()
            
            
            
        t4 = time.time()
        print(f'Plot time elapsed: {t4-t3:.2f} seconds')
            

        
            
            
            
        # e. print solution 
        
        if self.distribution_method != 'none':
            print("\n-----------------------------------------")
            print("Stationary Equilibrium Solution")
            print("-----------------------------------------")
            print(f"Steady State Assets = {self.a_ss:.2f}")
        
        if self.simulate or self.distribution_method == 'monte carlo' or self.full_euler_error:
            print("\n-----------------------------------------")
            print("Log10 Euler Equation Error Evaluation")
            print("-----------------------------------------")
            
            if self.full_euler_error:
                print(f"\nFull Grid Evalulation: Max Error  = {self.max_error:.2f}")
                print(f"Full Grid Evalulation: Average Error = {self.avg_error:.2f}")
        
            if self.simulate or self.distribution_method == 'monte carlo':
                print(f"\nSimulation: Max Error  = {self.max_error_sim:.2f}")
                print(f"Simulation: Average Error = {self.avg_error_sim:.2f}")
        
        
        t5 = time.time()
        print(f'\nTotal Run Time: {t5-t0:.2f} seconds')


        


###############################
# II. JIT Compiled Functions  #
###############################


#########################
# 1. Helper Functions  #
########################

@njit
def utility(c, sigma):
    """
    CRRA utility function.

    *Input 
        - c : Consumption
        - sigma: Risk aversion coefficient

    *Output
        - Utility value
    """
    
    eps = 1e-8
    
    if  sigma == 1:
        return np.log(np.fmax(c, eps))
    else:
        return (np.fmax(c, eps) ** (1 - sigma) -1) / (1 - sigma)

@njit
def u_prime(c, sigma) :
    """
    First order derivative of the CRRA utility function.

    *Input 
        - c : Consumption
        - sigma: Risk aversion coefficient

    *Output
        - Utility value
    """

    eps = 1e-8
    
    return np.fmax(c, eps) ** (-sigma)
    




################################################
# 2. Household and Policy Function Iteration  #
###############################################

@njit
def solve_hh(params_pfi):
        """
        Solves the household problem using policy function iteration on the euler equation.
        
        *Input
            - params_pfi: model parameters
        
        *Output
            -- pol_sav: the a' (savings) policy function
            -- pol_cons: the consumption policy function
            -- it: number of iterations
        """
        
        
        # a. Initialize
        
        r, w, beta, pi, grid_a, grid_z, sigma, maxit, tol = params_pfi
        
        Na = len(grid_a)
        Nz = len(grid_z)
        
        pol_sav_old    = np.zeros((Nz,Na)) #initial guess -- save nothing
        pol_sav = np.zeros((Nz,Na))            #savings policy function a'(z,a)
        pol_cons = np.zeros((Nz,Na))      #consumption policy function c(z,a)
        
        #alternative initil guess -- save everything
        #pol_sav_old[0,:] = (1+r)*grid_a + w*grid_z[0] 
        #pol_sav_old[1,:] = (1+r)*grid_a + w*grid_z[1] 
        
        # b. Iterate
        for it in range(maxit) :
            for i_z, z in enumerate(grid_z):        # current assets
                for i_a, a in enumerate(grid_a):    # current income shock
                
                
                    # i. next period assets bounds
                    lb_aplus = grid_a[0]                   # lower bound
                    ub_aplus = (1+r)*a + w*z                   # upper bound
                    
                    
                    # ii. set parameters for euler_eq_residual function
                    params_eer = a, z, pol_sav_old, i_z , r, w, beta, sigma, pi, grid_z, grid_a
                    
                    
                    # iii. use the sign of the euler equation to determine whether there is a corner or interior solution at the evaluated grid points
                    eulersign_lb = np.sign(euler_eq_residual(lb_aplus, params_eer))
                
                    #liquidity constrained, euler equation holds with positive inequality
                    if eulersign_lb == 1 :        
                        pol_sav[i_z, i_a] = lb_aplus
                
                    #interior solution, euler equation holds with negative inequality or equals zero
                    else:
                        
                        # check for errors 
                        eulersign_ub = np.sign( euler_eq_residual(ub_aplus, params_eer) )
                        
                        if eulersign_lb*eulersign_ub == 1:
                            raise Exception('Sign of lower bound and upperbound are the same - no solution to Euler Equation.')
                        
                        #find the root of the Euler Equation
                        pol_sav[i_z, i_a] = qe.optimize.root_finding.brentq( euler_eq_residual, lb_aplus, ub_aplus, args=(params_eer,) )[0]
                        
                # obtain consumption policy function
                pol_cons[i_z,:] = (1+r)*grid_a + w*grid_z[i_z] - pol_sav[i_z,:]
                
                
            # iv. calculate supremum norm
            dist = np.abs(pol_sav-pol_sav_old).max()
            
            if dist < tol :
                break
            
            pol_sav_old = np.copy(pol_sav)
    
    
    
        return pol_sav, pol_cons, it



@njit
def euler_eq_residual(a_plus, params_eer):
    """
    Returns the difference between the LHS and RHS of the Euler Equation.
    
    *Input
        - a_plus : current savings

    *Output
        - Returns euler equation residual
    """
    
    # a. Initialize
    a, z, pol_sav_old, i_z , r, w, beta, sigma, pi, grid_z, grid_a = params_eer
    
    Nz = len(grid_z)
    avg_marg_u_plus = 0
    
    # b. current consumption
    c = (1+r)*a + w*z - a_plus
    
    # c. expected marginal utility from consumption next period
    for i_zz in prange(Nz):
 
        # i. consumption next period
        c_plus = (1+r)*a_plus + w*grid_z[i_zz] - interp(grid_a, pol_sav_old[i_zz, :], a_plus)
 
        # ii. marginal utility next period
        marg_u_plus = u_prime(c_plus, sigma)
 
        # iii. calculate expected marginal utility
        weight = pi[i_z, i_zz]
 
        avg_marg_u_plus += weight * marg_u_plus
        
    # d. RHS of the euler equation
    ee_rhs = (1 + r) * beta * avg_marg_u_plus  
    
    return u_prime(c, sigma) - ee_rhs



####################
# 3. Simulation   #
##################

@njit(parallel=True)
def simulate_MarkovChain(pol_cons, pol_sav, params_sim):
    """
    Simulates markov chain for T periods for N households and calculates the euler equation error for all 
    individuals at each point in time. In addition, it checks the grid size by issuing a warning if 1% of 
    households are at the maximum value (right edge) of the grid. The algorithm is from Ch.7 in Heer and Maussner.
    
    *Input
        - pol_cons: consumption policy function 
        - pol_sav: savings policy function 
        - params_sim: model parameters
    
    *Output
        - sim_c: consumption profile
        - sim_sav: savings (a') profile
        - sim_z: Income shock profile.
        - sim_m: cash-on-hand profile ((1+r)a + w*z)
        - euler_error_sim : error when the euler equation equality holds
    """
    
    # 1. initialization
    
    a0, z0, r, w, simN, simT, grid_z, grid_a, sigma, beta, pi, shock_history = params_sim
    
    sim_sav = np.zeros((simT,simN))
    sim_c = np.zeros((simT,simN))
    sim_m = np.zeros((simT,simN))
    sim_z = np.zeros((simT,simN), np.float64)
    sim_z_idx = np.zeros((simT,simN), np.int32)
    edge = 0
    euler_error_sim = np.empty((simT,simN)) * np.nan
    
    
    
    # 2. helper functions
    
    # savings policy function interpolant
    polsav_interp = lambda a, z: interp(grid_a, pol_sav[z, :], a)
    
    # marginal utility
    u_prime = lambda c : c**(-sigma)
    
    #inverse marginal utility
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
            sim_z_idx[t,i] = shock_history[t,i]
            sim_z[t,i] = grid_z[sim_z_idx[t,i]]
                
            # c. income
            y = w*sim_z[t,i]
            
            # d. cash-on-hand path
            sim_m[t, i] = (1 + r) * a_lag + y
            
            # e. savings path
            sim_sav[t,i] = polsav_interp(a_lag,sim_z_idx[t,i])
            if sim_sav[t,i] < grid_a[0] : sim_sav[t,i] = grid_a[0]     #ensure constraint binds
            
            # f. consumption path
            
            sim_c[t,i] = sim_m[t, i] - sim_sav[t,i]   
            
           
                
            # g. error evaluation
            
            check_out=False
            if sim_sav[t,i] == pol_sav[sim_z_idx[t,i],-1]:
                edge = edge + 1
                check_out=True
                
            constrained=False
            if sim_sav[t,i] == grid_a[0]:
                constrained=True
            
                
            if sim_c[t,i] < sim_m[t,i] and constrained==False and check_out==False :
                
                avg_marg_c_plus = 0
                
                for i_zz in range(len(grid_z)):      #next period productivity
                
                    sav_int = polsav_interp(sim_sav[t,i],i_zz)
                    if sav_int < grid_a[0] : sav_int = grid_a[0]     #ensure constraint binds
                
                    c_plus = (1 + r) * sim_sav[t,i] + w*grid_z[i_zz] - polsav_interp(sim_sav[t,i],i_zz)
                        
                    #expectation of marginal utility of consumption
                    avg_marg_c_plus += pi[sim_z_idx[t,i],i_zz] * u_prime(c_plus)
                
                #euler error
                euler_error_sim[t,i] = 1 - (u_prime_inv(beta*(1+r)*avg_marg_c_plus) / sim_c[t,i])
            
            
    # 4. transform euler error to log_10 and get max and average
    euler_error_sim = np.log10(np.abs(euler_error_sim))
                
    
    # 5. grid size evaluation
    frac_outside = edge/grid_a.size
    if frac_outside > 0.01 :
        raise Exception('Increase grid size!')
    
    
    
    return sim_c, sim_sav, sim_z, sim_m, euler_error_sim



###############################################################################
# 4. Stationary Distribution: Discrete Approximation and Forward Iteration   #
##############################################################################

@njit
def discrete_stationary_density(pol_sav, params_discrete):
    """
    Discrete approximation of the density function. Approximates the stationary joint density through forward 
    iteration and linear interpolation over a discretized state space. By default the code uses a finer grid than 
    the one in the solution but one could use the same grid here. The algorithm is from Ch.7 in Heer and Maussner.
    
    *Input
        - pol_sav: savings policy function
        - params_discrete: model parameters
        
    *Output
        - stationary_pdf: joint stationary density function
        - it: number of iterations
    """
    
    # a. initialize
    
    grid_a, grid_a_fine, Nz, pi, pi_stat, maxit, tol = params_discrete
    
    Na_fine = len(grid_a_fine)
    
    # initial guess uniform distribution
    stationary_pdf_old = np.ones((Na_fine, Nz))/Na_fine
    stationary_pdf_old = stationary_pdf_old * np.transpose(pi_stat)
    stationary_pdf_old = stationary_pdf_old.T
    
    # b. fixed point iteration
    for it in range(maxit):   # iteration 
        
        stationary_pdf = np.zeros((Nz, Na_fine))    # distribution in period t+1
             
        for iz in range(Nz):     # iteration over productivity types in period t
            
            for ia, a0 in enumerate(grid_a_fine):  # iteration over assets in period t   
            
                # i. interpolate 
                
                a_intp = interp(grid_a, pol_sav[iz,:], a0) # linear interpolation for a'(z, a)    
                
                # ii. obtain distribution in period t+1   
                # ii. obtain distribution in period t+1   
                
                #left edge of the grid
                if a_intp <= grid_a_fine[0]:
                    for izz in range(Nz):
                        stationary_pdf[izz,0] = stationary_pdf[izz,0] + stationary_pdf_old[iz,ia]*pi[iz,izz]
                        
                
                #right edge of the grid
                elif a_intp >= grid_a_fine[-1]:
                    for izz in range(Nz):
                        stationary_pdf[izz,-1] = stationary_pdf[izz,-1] + stationary_pdf_old[iz,ia]*pi[iz,izz]
                        
                    
                #inside the grid range, linearly interpolate
                else:
                    
                    j = np.sum(grid_a_fine <= a_intp) # a_intp lies between grid_sav_fine[j-1] and grid_sav_fine[j]
                    
                    p0 = (a_intp-grid_a_fine[j-1]) / (grid_a_fine[j]-grid_a_fine[j-1])
                    
                    for izz in range(Nz):
                    
                        stationary_pdf[izz,j] = stationary_pdf[izz,j] + p0*stationary_pdf_old[iz,ia]*pi[iz,izz]
                        stationary_pdf[izz,j-1] =stationary_pdf[izz,j-1] + (1-p0)*stationary_pdf_old[iz,ia]*pi[iz,izz]
                
        
        
        #stationary distribution by percent 
        stationary_pdf=stationary_pdf/np.sum(np.sum(stationary_pdf,axis=0)) 
        
        # iii. calculate supremum norm
        dist = np.abs(stationary_pdf-stationary_pdf_old).max()
        
        if dist < tol:
            break
        
        else:
            stationary_pdf_old = np.copy(stationary_pdf)
        
    return stationary_pdf, it




# run model
cs_pfi2=ConSavePFI()
cs_pfi2.solve_model()
