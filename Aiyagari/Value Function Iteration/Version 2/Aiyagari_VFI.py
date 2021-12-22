
"""
Author: Jacob Hess 
First Version: December 2020
This Version: December 2021

Description: Finds the stationary equilibrium for the infinitely lived household in a production economy with incomplete 
markets and idiosyncratic income risk as in Aiyagari (1994) using discrete value function iteration. The continuous income process 
is discretely approximated using the Rouwenhorst method.

To find the stationary distribution one can choose from three methods: 

1) Discrete approximation of the density function which conducts a fixed point iteration with linear interpolation
2) Eigenvector method to solve for the exact stationary density.
3) Monte carlo simulation. 

Finally, to evaluate the accuracy of the solution the code computes the euler equation error with two different methods. 
One is by calculating the error across the entire in the state space. The other is through the monte carlo simulation by 
calulating the error for each individual. 

Aknowledgements: I wrote the algorithms using the following resources :
    1) Gianluca Violante's global methods and distribution approximation notes (https://sites.google.com/a/nyu.edu/glviolante/teaching/quantmacro)
    2) Heer and Maussner 2nd ed. Ch. 7
    3) Raul Santaeulalia-Llopis' ABHI Notes (http://r-santaeulalia.net/)
    4) Alexander Ludwig's notes (https://alexander-ludwig.com/)
    5) Jeppe Druedahl (https://github.com/JeppeDruedahl) and NumEconCopenhagen (https://github.com/NumEconCopenhagen)
    
Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual)
    -- Interpolation from EconForge
       * optimized interpolation routines for python/numba
       * to install 'conda install -c conda-forge interpolation'
       * https://github.com/EconForge/interpolation.py

Note 1: If simulation tells you to increase grid size, increase self.a_max in function setup_parameters.
Note 2: Be advised that by default there are many grid points which makes the eigenvector method quite slow (calculation up to 20 minutes using
grid_a, 75 minutes using grid_a_fine). This is due to numpy.linalg.eig.
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

class AiyagariVFI:
    
    """
    Class object of the model. AiyagariVFI().solve_model() runs everything
    """

    ############
    # 1. setup #
    ############

    def __init__(self, a_bar = 0,              #select borrowing limit
                       plott = 1,               #select 1 to make plots
                       full_euler_error = 0,        #select to compute euler_error for entire state space
                       distribution_method = 'discrete', #Approximation method of the stationary distribution. 
                                                       #Options: 'discrete', 'eigenvector', 'monte carlo' 
                       plot_supply_demand = 0 # select 1 for capital market supply/demand graph
                       ):
        
        #parameters subject to changes
        self.a_bar, self.plott, self.full_euler_error = a_bar, plott, full_euler_error
        self.distribution_method, self.plot_supply_demand  = distribution_method, plot_supply_demand 
        
        self.setup_parameters()
        self.setup_grid()
        self.setup_discretization()
        
        #pack parameters for jitted functions
        
        self.params_vfi = self.beta, self.pi, self.grid_a, self.grid_z, self.sigma, self.maxit_hh, self.tol_hh
        
        if distribution_method == 'discrete' or self.plot_supply_demand:
            self.params_discrete = self.grid_a, self.grid_a_fine, self.Nz, self.pi, self.pi_stat, self.maxit_dis, self.tol_dis
            
        if self.distribution_method == 'monte carlo':
            self.params_sim = self.a0, self.simN, self.simT, self.grid_z, self.grid_a, self.sigma, self.beta, self.pi, self.shock_history
        
        #warnings 
        
        if self.distribution_method != 'discrete' and self.distribution_method != 'eigenvector' and self.distribution_method != 'monte carlo':
            raise Exception("Stationary distribution approximation method incorrectly entered: Choose 'discrete', 'eigenvector' or 'monte carlo' ")
            
        if self.plott != 1 and self.plott != 0:
            raise Exception("Plot option incorrectly entered: Choose either 1 or 0.")
            
        if self.full_euler_error != 1 and self.full_euler_error != 0:
            raise Exception("Euler error full grid evaluation option incorrectly entered: Choose either 1 or 0.")
        
        
        
        

    def setup_parameters(self):
    
       # a. model parameters
       self.sigma=2      #crra coefficient
       self.beta = 0.96  # discount factor
       self.rho = (1-self.beta)/self.beta #discount rate
       self.delta = 0.08  # depreciation rate
       self.alpha = 0.36  # cobb-douglas coeffient
    
       # b. hh solution
       self.tol_hh = 1e-6  # tolerance for policy function iterations
       self.maxit_hh = 2000  # maximum number of iterations when finding policy function in hh problem
      
       # AR(1) income process
       self.Nz = 7                 #number of discrete income states
       self.z_bar = 0             #constant term in continuous income process (not the mean of the process)
       self.rho_z = 0.9    #autocorrelation coefficient
       self.sigma_z = 0.2         #std. dev. of income process at annual frequency
    
       # asset grid 
       self.Na = 1000
       self.a_min = self.a_bar
       self.a_max = 200
       self.curv = 3 
       
       if self.distribution_method == 'discrete' or self.distribution_method == 'eigenvector' or self.full_euler_error :
           self.Na_fine = self.Na*3
    
       # c. stationary distribution 
       
       if self.distribution_method == 'discrete' or self.graph_supply_demand:
           self.tol_dis = 1e-6
           self.maxit_dis = 2000
       
       if self.distribution_method == 'monte carlo':
           self.seed = 123
           self.simN = 50_000  # number of households
           self.simT =  2000 # number of time periods to simulate
           self.sim_burnin = 1000  # burn-in periods before calculating average savings
           self.init_asset = 1.0  # initial asset (homogenous)
           self.a0 = self.init_asset * np.ones(self.simN)  #initial asset for all individuals
    
       # d. steady state solution
       self.ss_r_tol = 1e-4  # tolerance for finding interest rate
       self.dp_big = 1/10      # dampening parameter to update new interest rate guess 
       self.dp_small = 1/100    # dampening parameter to prevent divergence
       self.maxit_ss = 100    # maximum iterations steady state solution
       
       # e. complete markets solution
       self.r_cm = 1/self.beta - 1
       self.k_cm = self.k_demand(self.r_cm)



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
        self.pi_stat = self.mc.stationary_distributions.T.ravel()   #ravel to resize array from 2d to 1d for rv_discrete
        self.grid_z = np.exp(self.mc.state_values)
        
        avg_z = np.sum(self.grid_z * self.pi_stat)
        self.grid_z = self.grid_z / avg_z  # force mean one
        
        # c. income shock sequence for each individual for simulation
        if self.distribution_method == 'monte carlo':
         
            # draw income shocks for each individual
            self.shock_history= np.zeros((self.simT, self.simN))
            
            # initial income shock drawn for each individual from initial distribution
            random_z = rv_discrete(values=(np.arange(self.Nz),self.pi_stat), seed=self.seed)
            z0_idx = random_z.rvs(size=self.simN)
            
            for n in range(self.simN) :
                seed_sim = self.seed + n 
                
                self.shock_history[:,n] = self.mc.simulate_indices(self.simT, init=z0_idx[n], random_state=seed_sim)       
        
    
    
    
    
    #######################
    # 2. Helper Functions #
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
    
    def r_func(self, k):
        return  self.f_prime(k) - self.delta

    def w_func(self, ret):
        k = self.f_prime_inv(ret + self.delta)
        return self.f(k) - self.f_prime(k) * k
    
    def k_demand(self,ret):
        return (self.alpha/(ret+self.delta))**(1/(1-self.alpha))
    
    


    
    ####################################################
    # 3. Stationary Distribution: Eigenvector Method   #
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
                
                #take the grid index to the right. a_intp lies between grid_a_fine[j-1] and grid_a_fine[j]. 
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

    



    ######################################
    # 4. Euler Equation Error Analysis  #
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
                    c = (1 + self.r_ss) * a + self.w_ss * z - a_plus
                    avg_marg_c_plus = 0
                    
                    # iii. expected marginal utility
                    for i_zz, z_plus in enumerate(self.grid_z):      #next period productivity
                    
                        c_plus = (1 + self.r_ss) * a_plus + self.w_ss * z_plus - interp(self.grid_a, self.pol_sav[i_zz,:], a_plus)
                        
                        #expectation of marginal utility of consumption
                        avg_marg_c_plus += self.pi[i_z,i_zz] * u_prime(c_plus)
                    
                    # iv. compute euler error
                    euler_error[i_z, i_a] = 1 - u_prime_inv(self.beta*(1+self.r_ss)*avg_marg_c_plus) / c
                    
       
        # ii. transform euler error with log_10. take max and average
        euler_error = np.log10(np.abs(euler_error))
        max_error =  np.nanmax(np.nanmax(euler_error, axis=1))
        avg_error = np.nanmean(euler_error) 
        
        
        
        return euler_error, max_error, avg_error





    ###############################
    # 5. Stationary Equilibrium   #
    ###############################
    
    def graph_supply_demand(self):
        
        """
        Plots capital market supply and demand.
        
        *Output
            - k_demand : capital demand as a function of the interest rate
            - k_supply : capital supply as a function of the interest rate
        """
        
        #a. initialize
        k_demand = np.empty(self.r_vec.size)
        k_supply = np.empty(self.r_vec.size)
        
        for idx, r_graph in enumerate(self.r_vec):
            
            # b. capital demand
            k_demand[idx] = self.k_demand(r_graph)
            
            # c. capital supply
            w_graph = self.w_func(r_graph)
            
            # d. solve economy
            # i. household
            _, pol_sav_graph, _, _ = solve_hh(self.params_vfi, r_graph, w_graph)
            
            # ii. aggregation 
            
            #stationary_pdf_graph, _ = self.eigen_stationary_density()
            stationary_pdf_graph, _ = discrete_stationary_density(pol_sav_graph, self.params_discrete)
        
            # aggregate capital stock
            k_ss_graph = np.sum(np.dot(stationary_pdf_graph, self.grid_a_fine))
            
            k_supply[idx] = np.mean(k_ss_graph)
        
        return k_demand, k_supply
            
    

    def ge_algorithm(self, r_ss_guess):
        
        """
        General equilibrium solution algorithm.
        """
        
        #given r_ss_guess as the guess for the interest rate (step 1)
        
        # a. obtain prices from firm FOCs (step 2)
        r_ss = np.copy(r_ss_guess)
        w_ss = self.w_func(r_ss)



        # b. solve the HH problem (step 3)
        
        print('\nSolving household problem...')
        
        t1 = time.time()
        
        self.VF, self.pol_sav, self.pol_cons, self.it_hh = solve_hh(self.params_vfi, r_ss, w_ss)
        
        if self.it_hh < self.maxit_hh-1:
            print(f"Value function convergence in {self.it_hh} iterations.")
        else : 
            raise Exception("No value function convergence.")

            
        t2 = time.time()
        print(f'Household problem time elapsed: {t2-t1:.2f} seconds')
        
        
        
        # c. stationary distribution (step 4)
        
        # discrete approximation
        if self.distribution_method == 'discrete':
            
            print("\nStationary Distribution Solution Method: Discrete Approximation and Forward Iteration on Density Function")
            print("\nComputing...")
            
            # i. approximate stationary density
            self.stationary_pdf, self.it_pdf = discrete_stationary_density(self.pol_sav, self.params_discrete)
            
            if self.it_pdf < self.maxit_dis-1:
                print(f"Convergence in {self.it_pdf} iterations.")
            else : 
                raise Exception("No density function convergence.")
            
            # ii. steady state assets
            self.k_ss = np.sum(np.dot(self.stationary_pdf, self.grid_a_fine))
            
            # iii. marginal wealth density
            self.stationary_wealth_pdf = np.sum(self.stationary_pdf, axis=0)
            
            t3 = time.time()
            print(f'Density approximation time elapsed: {t3-t2:.2f} seconds')
        
        
        
        # eigenvector
        if self.distribution_method == 'eigenvector':
            
            print("\nStationary Distribution Solution Method: Eigenvector Method for Exact Stationary Density")
            print("\nComputing...")
            
            self.stationary_pdf, self.Q = self.eigen_stationary_density()
        
            # i. aggregate capital stock
            self.k_ss = np.sum(np.dot(self.stationary_pdf, self.grid_a_fine))
            
            # iii. marginal wealth density
            self.stationary_wealth_pdf = np.sum(self.stationary_pdf, axis=0)
            
            t3 = time.time()
            print(f'Density computation time elapsed: {t3-t2:.2f} seconds')
        
        
        
        # monte carlo simulation
        if self.distribution_method == 'monte carlo':
            
            print("\nStationary Distribution Solution Method: Monte Carlo Simulation")
            
            print("\nSimulating...")
            
            # i. simulate markov chain and endog. variables 
            self.ss_sim_k, self.ss_sim_sav, self.ss_sim_z, self.ss_sim_c, self.ss_sim_m, self.euler_error_sim = simulate_MonteCarlo(
                self.pol_cons,
                self.pol_sav,
                r_ss,
                w_ss,
                self.params_sim
            )
            
            # ii. steady state assets
            self.k_ss = np.mean(self.ss_sim_k[self.sim_burnin:])
            
            # iii. max and average euler error error, ignores nan which is when the euler equation does not bind
            self.max_error_sim =  np.nanmax(self.euler_error_sim)
            self.avg_error_sim = np.nanmean(np.nanmean(self.euler_error_sim)) 
            
            t3 = time.time()
            print(f'Simulation time elapsed: {t3-t2:.2f} seconds')
        
        

        # d. calculate interest rate difference
        r_ss_new = self.r_func(self.k_ss)
        diff = r_ss_guess - r_ss_new
        
        return diff





    ######################
    # 6. Main Function   #
    ######################

    def solve_model(self):
    
        """
        Finds the stationary equilibrium.
        """    
        
        t0 = time.time()    #start the clock
    
        # a. initial interest rate guess (step 1)
        r_guess = 0.02       
        
        # We need (1+r)beta < 1 for convergence.
        assert (1 + r_guess) * self.beta < 1, "Stability condition violated."
        
            
            
        # b. iteration to find equilibrium interest rate r_ss
        
        for it in range(self.maxit_ss) :
            
            print("\n-----------------------------------------")
            print("Iteration #"+str(it+1))
            
            diff_old=np.inf
            diff = self.ge_algorithm(r_guess)
            
            if abs(diff) < self.ss_r_tol :
                print("\n-----------------------------------------")
                print('\nConvergence!')
                break
            else :
                #update guess with adaptive dampening 
                if np.abs(diff) > np.abs(diff_old):
                    r_guess = r_guess - self.dp_small*diff  #to prevent divergence force a conservative new guess
                else:
                    r_guess = r_guess - self.dp_big*diff
                
                print(f"\nNew interest rate guess = {r_guess:.5f} \t diff = {diff:8.5f}")
                diff_old=np.copy(diff)
        
        if it > self.maxit_ss-1 :
            print("No convergence")
            
        #stationary equilibrium prices and precautionary savings rate
        self.r_ss = np.copy(r_guess)
        self.w_ss = self.w_func(self.r_ss)
        self.precaution_save = self.r_cm - self.r_ss
        
        t4 = time.time()
        print('Total iteration time elapsed: '+str(time.strftime("%M:%S",time.gmtime(t4-t0))))
        
        
        
        # c. calculate euler equation error across the state space
    
        if self.full_euler_error:
            print("\nCalculating Euler Equation Error...")
            
            self.euler_error, self.max_error, self.avg_error = self.ee_error()
            
            t5 = time.time()
            print(f'Euler Eq. error calculation time elapsed: {t5-t4:.2f} seconds')
            
        else: 
            t5 = time.time()
        
        
        
        # d. plot
    
        if self.plott:
            
            print('\nPlotting...')
        
             ##### Solutions #####
            plt.plot(self.grid_a, self.VF.T)
            plt.title('Value Function')
            plt.xlabel('Assets')
            #plt.savefig('value_function_vfi_aiyagari.pdf')
            plt.show()
            
            plt.plot(self.grid_a, self.pol_sav.T)
            plt.title("Savings Policy Function")
            plt.xlabel('Assets')
            #plt.savefig('savings_policyfunction_vfi_aiyagari.pdf')
            plt.show()
            
            plt.plot(self.grid_a, self.pol_cons.T)
            plt.title("Consumption Policy Function")
            plt.xlabel('Assets')
            #plt.savefig('consumption_policyfunction_vfi_aiyagari.pdf')
            plt.show()
            
            if self.full_euler_error:
                plt.plot(self.grid_a_fine, self.euler_error.T)
                plt.title('Log10 Euler Equation Error')
                plt.xlabel('Assets')
                #plt.savefig('log10_euler_error_vfi_aiyagari.pdf')
                plt.show()
                
            if self.plot_supply_demand:
                print('\nPlotting supply and demand...')
                
                self.r_vec = np.linspace(-0.01,self.rho-0.001,25)
                self.k_demand, self.k_supply = self.graph_supply_demand()
            
                plt.plot(self.k_demand,self.r_vec)
                plt.plot(self.k_supply,self.r_vec)
                plt.plot(self.k_supply,np.ones(self.r_vec.size)*self.rho,'--')
                plt.title('Capital Market')
                plt.legend(['Demand','Supply','Supply in CM'])
                plt.xlabel('Capital')
                plt.ylabel('Interest Rate')
                #plt.savefig('capital_supply_demand_aiyagari.pdf')
                plt.show()
                
                
                
            ##### Distributions ####
            if self.distribution_method == 'discrete' or self.distribution_method == 'eigenvector':
                
                # marginal wealth density
                plt.plot(self.grid_a_fine, self.stationary_wealth_pdf)
                plt.title("Stationary Wealth Density (Discrete Approx.)") if self.distribution_method == 'discrete' else plt.title("Stationary Wealth Density (Eigenvector Method)")
                plt.xlabel('Assets')
                #plt.savefig('wealth_density_vfi_aiyagari_discrete.pdf') if self.distribution_method == 'discrete' else plt.savefig('wealth_density_vfi_aiyagari_eigenvector.pdf')
                plt.show()
                
            
            
            if self.distribution_method == 'monte carlo':
                sns.histplot(self.ss_sim_sav, bins=100, stat='density')
                plt.title("Stationary Wealth Density (Monte Carlo Approx.)")
                plt.xlabel('Assets')
                #plt.savefig('wealth_density_vfi_aiyagari_montecarlo.pdf')
                plt.show()
        

        t6 = time.time()
        print(f'Plot time elapsed: {t6-t5:.2f} seconds')
        
        
        
        # e. print solution 
        
        print("\n-----------------------------------------")
        print("Stationary Equilibrium Solution")
        print("-----------------------------------------")
        print(f"Steady State Interest Rate = {r_guess:.5f}")
        print(f"Steady State Capital = {self.k_ss:.2f}")
        print(f"\nPrecautionary Savings Rate = {self.precaution_save:.5f}")
        print(f"Capital stock in incomplete markets is {((self.k_ss - self.k_cm)/self.k_cm)*100:.2f} percent higher than with complete markets")
        print('\nTotal run time: '+str(time.strftime("%M:%S",time.gmtime(t6-t0))))
        
        if self.distribution_method == 'monte carlo' or self.full_euler_error:
            print("\n-----------------------------------------")
            print("Log10 Euler Equation Error Evaluation")
            print("-----------------------------------------")
        
        if self.full_euler_error:
            print(f"\nFull Grid Evalulation: Max Error  = {self.max_error:.2f}")
            print(f"Full Grid Evalulation: Average Error = {self.avg_error:.2f}")
    
        if self.distribution_method == 'monte carlo':
            print(f"\nMonte Carlo Simulation: Max Error  = {self.max_error_sim:.2f}")
            print(f"Monte Carlo Simulation: Average Error = {self.avg_error_sim:.2f}")





################################
# II. JIT Compiled Functions  #
##############################


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



###########################################
# 2. Household Value Function Iteration  #
##########################################

@njit(parallel=True)
def solve_hh(params_vfi, r, w):
    """
    Solves the household problem by value function iteration.
    
    *Input
        - params_vfi: model parameters
        - r : interest rate 
        - w : wage
    
    *Output
        -- VF: value function
        -- pol_sav: the a' (savings) policy function
        -- pol_cons: the consumption policy function
        -- it: number of iterations
    """
    

    # a. Initialize
    beta, pi, grid_a, grid_z, sigma, maxit, tol = params_vfi
    
    Nz = len(grid_z)
    Na = len(grid_a)
    
    VF_old    = np.zeros((Nz,Na))  #initial guess
    VF = np.copy(VF_old)            #contracted value function aka Tv
    pol_sav = np.copy(VF_old)       #savings policy function a'(z,a)
    pol_cons = np.copy(VF_old)      #consumption policy function c(z,a)
    
    # b. Iterate
    for it in range(maxit) :
        for iz in range(Nz):
            for ia in prange(Na):
                
                # i. budget constraint 
                c = (1+r)*grid_a[ia] + w*grid_z[iz] - grid_a
                
                # ii. utility and impose nonnegativity for consumption
                util = utility(c, sigma)
                util[c < 0] = -10e10
                
                # iii. value and policy functions
                RHS = util + beta*(np.dot(pi[iz,:], VF_old))    #RHS of Bellman
                
                VF[iz,ia] = np.max(RHS)     #take maximum value for value function
                
                pol_sav[iz,ia] = grid_a[np.argmax(RHS)]      #policy function for how much to save
            
            # obtain consumption policy function
            pol_cons[iz,:] = (1+r)*grid_a + w*grid_z[iz] - pol_sav[iz,:]
        
        # iv. calculate supremum norm
        dist = np.abs(VF-VF_old).max()
        
        if dist < tol :
            break
        
        VF_old = np.copy(VF)

    
    return VF, pol_sav, pol_cons, it



#########################################################
# 3. Stationary Distribution: Monte Carlo Simulation   #
########################################################

@njit(parallel=True)
def simulate_MonteCarlo(pol_cons, pol_sav, r, w, params_sim):
    
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
    
    # a. initialization
    a0, simN, simT, grid_z, grid_a, sigma, beta, pi, shock_history = params_sim
    
    sim_sav = np.zeros(simN)
    sim_c = np.zeros(simN)
    sim_m = np.zeros(simN)
    sim_z = np.zeros(simN)
    sim_z_idx = np.zeros(simN, np.int32)
    sim_k = np.zeros(simT)
    euler_error_sim = np.empty(simN) * np.nan
    edge = 0
    
    # b. helper functions
    
    # savings policy function interpolant
    polsav_interp = lambda a, z: interp(grid_a, pol_sav[z, :], a)
    
    # marginal utility
    u_prime = lambda c : c**(-sigma)
    
    #inverse marginal utility
    u_prime_inv = lambda x : x ** (-1/sigma)
    
    # c. simulate markov chain
    for t in range(simT):   #time

        draw = np.linspace(0, 1, simN)
        np.random.shuffle(draw)
        
        #calculate cross-sectional moments for agg. capital
        if t <= 0:
            sim_k[t] = np.mean(a0)
        else:
            sim_k[t] = np.mean(sim_sav)
        
        for i in prange(simN):  #individual

             # i. states 
            if t == 0:
                a_lag = a0[i]
                
            else:
                a_lag = sim_sav[i]
                
            # ii. shock realization.
            sim_z_idx[i] = shock_history[t,i]
            sim_z[i] = grid_z[sim_z_idx[i]]
                
            # iii. income
            y = w*sim_z[i]
            
            # iv. cash-on-hand
            sim_m[i] = (1 + r) * a_lag + y
            
            # v. savings path
            sim_sav[i] = polsav_interp(a_lag, sim_z_idx[i])
            if sim_sav[i] < grid_a[0] : sim_sav[i] = grid_a[0]     #ensure constraint binds
            
            # vi. consumption path
            
            sim_c[i] = sim_m[i] - sim_sav[i] 
            
            # vii. error evaluation
            
            check_out=False
            if sim_sav[i] >= pol_sav[sim_z_idx[i],-1]:
                edge = edge + 1
                check_out=True
                
            constrained=False
            if sim_sav[i] == grid_a[0]:
                constrained=True
            
                
            if sim_c[i] < sim_m[i] and constrained==False and check_out==False :
                
                avg_marg_c_plus = 0
                
                for i_zz in range(len(grid_z)):      #next period productivity
                    
                    sav_int = polsav_interp(sim_sav[i],i_zz)
                    if sav_int < grid_a[0] : sav_int = grid_a[0]     #ensure constraint binds
                
                    c_plus = (1 + r) * sim_sav[i] + w*grid_z[i_zz] - sav_int
                        
                    #expectation of marginal utility of consumption
                    avg_marg_c_plus += pi[sim_z_idx[i],i_zz] * u_prime(c_plus)
                
                #euler error
                euler_error_sim[i] = 1 - (u_prime_inv(beta*(1+r)*avg_marg_c_plus) / sim_c[i])
            
            
            
    # d. transform euler eerror to log_10 and get max and average
    euler_error_sim = np.log10(np.abs(euler_error_sim))
                
    # e. grid size evaluation
    frac_outside = edge/grid_a.size
    if frac_outside > 0.01 :
        raise Exception('Increase grid size!')

    return sim_k, sim_sav, sim_z, sim_c, sim_m, euler_error_sim





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
                    
                    j = np.sum(grid_a_fine <= a_intp) # a_intp lies between grid_a_fine[j-1] and grid_a_fine[j]
                    
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






#run everything

ge_vfi = AiyagariVFI()
ge_vfi.solve_model()
