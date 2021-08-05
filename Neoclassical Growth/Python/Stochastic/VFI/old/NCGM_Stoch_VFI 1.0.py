"""
Author: Jacob Hess
Date: July 2021

Description: This code solves the social planner's problem for stochastic neoclassical growth model. Since the 
welfare theorems hold the solution coincides with the competitive equilibrium solution. The model is solved using 
value function iteration and simulates a markov chain. For the simulation the code interpolates the next period 
capital policy function by either cubic interpolation or by approximating it with chebyshev polynomials and OLS. 
I also calculate the euler equation errors to assess the accuracy of the solution two different ways. One is
by calculating the error for each period in the simulation and the other is by calculating the error over the 
entire state space.

Acknowledgements:
    1) Heer and Maussner Ch. 4
    2) Fabrice Collard's "Value Iteration" notes (http://fabcol.free.fr/notes.html)
    3) Wouter den Haan's "accuracy tests" notes (https://www.wouterdenhaan.com/notes.htm)
    
Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual
    -- QuantEcon (to install: 'conda install quantecon')
"""



import time
import numpy as np
from numpy import linalg as la
from scipy import interpolate
from numba import njit, prange
import quantecon as qe
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')



#############
# I. Model  #
############


class ncgmVFI:
    
    ############
    # 1. setup #
    ############

    def __init__(self, plott = 1, simulate=1, sim_interpolate_type = 'chebyshev', full_euler_error = 1):              
            
            #parameters subject to changes
            self.plott = plott  #select 1 to make plots
            self.simulate = simulate  #select 1 to run simulation
            self.sim_interpolate_type = sim_interpolate_type  #interpolation during simulation. options: 'chebyshev, 'cubic'
            self.full_euler_error = full_euler_error  #select 1 to calculate euler error on entire grid
            
            self.setup_parameters()
            self.setup_grid()
            self.setup_markov()
        
            self.params = self.sigma, self.beta, self.delta, self.alpha, self.grid_k, self.grid_z, \
                self.Nz, self.Nk, self.pi, self.maxit, self.tol

    def setup_parameters(self):

        # a. model parameters
        self.sigma=2    #crra coefficient
        self.beta = 0.95  # discount factor
        self.rho = (1-self.beta)/self.beta #discount rate
        self.delta = 0.1  # depreciation rate
        self.alpha = 1/3  # cobb-douglas coeffient
        
        # b. tfp process parameters
        self.rho_z = 0.9           #autocorrelation coefficient
        self.sigma_z = 0.04        #std. dev. of shocks at annual frequency
        self.Nz = 7                #number of discrete income states
        self.z_bar = 0             #constant term in continuous productivity process (not the mean of the process)
        
        # c. steady state solution
        self.k_ss = (self.alpha/(1/self.beta - (1-self.delta)))**(1/(1 - self.alpha)) #capital
        self.c_ss = self.k_ss**(self.alpha) - self.delta*self.k_ss #consumption
        self.i_ss = self.delta*self.k_ss    #investment
        self.y_ss = self.k_ss**(self.alpha) #output

        # d. vfi solution
        self.tol = 1e-6  # tolerance for vfi
        self.maxit = 2000  # maximum number of iterations 

        # capital grid 
        self.Nk = 1000
        self.dev = 0.9
        self.k_min = (1-self.dev)*self.k_ss
        self.k_max = (1+self.dev)*self.k_ss
        
        # e. simulation
        self.seed = 123
        self.cheb_order = 10    #chebyshev polynomial order
        self.simT = 200    #number of simulation periods
        
        # f. euler equation error analysis
        if self.full_euler_error:
            self.Nk_fine=2500
        
    
    
    def setup_grid(self):
        
        # a. capital grid
        self.grid_k = np.linspace(self.k_min, self.k_max, self.Nk)
        
        # b. finer grids for euler error analysis
        if self.full_euler_error:
            self.grid_k_fine = np.linspace(self.k_min, self.k_max, self.Nk_fine)
        
    
    
    def setup_markov(self):
        
        # a. discretely approximate the continuous income process 
        self.mc = qe.markov.approximation.rouwenhorst(self.Nz, self.z_bar, self.sigma_z, self.rho_z)
        #self.mc = qe.markov.approximation.tauchen(self.rho_z, self.sigma_u, self.z_bar, 3, self.Nz)

        # b. transition matrix and states
        self.pi = self.mc.P
        self.grid_z = np.exp(self.mc.state_values)
        
        # c. simulation
        
        #generate markov simulation for tfp
        self.sim_z =  self.mc.simulate(self.simT, init=np.log(self.grid_z[int(self.Nz/2)]), num_reps=None, random_state=self.seed)
        self.sim_z = np.exp(self.sim_z)

        # indices of grid_z over the simulation 
        self.sim_z_idx = self.mc.simulate_indices(self.simT, init=int(self.Nz/2), num_reps=None, random_state=self.seed)

        

    ########################
    # 2. Helper Functions #
    #######################
    
    
    
    def grid_to_nodes(self, x, x_min, x_max):
        """
        Scales x \in [x_min,x_max] to z \in [-1,1]
        """    
        
        return (2 * (x - x_min) / (x_max - x_min)) - 1
    
    
        
    def chebyshev_polynomial(self, roots, n):
        """ 
        Constructs a psuedo-Vandermonde matrix of Chebyshev polynomial expansion using the recurrence relation
        
        This function is equivalent to np.polynomial.chebyshev.chebvander(roots, n-1)
        """
        T = np.zeros((np.size(roots),n))
    
        T[:,0] = np.ones((np.size(roots),1)).T
    
        T[:,1] = roots.T
    
        for i in range(1,n-1):
            T[:,i+1] = 2 * roots * T[:,i] - T[:,i-1]
            
        return T
    
    
    def u_prime(self, c) :
        eps = 1e-8
        
        return np.fmax(c, eps) ** (-self.sigma)

    
    def u_prime_inv(self, x):    
        eps = 1e-8
        
        return np.fmax(x, eps) ** (-1/self.sigma)



    ##################
    # 3. Simulation #
    #################
    
    
    
    def simulation(self):
        """
        Simulates the economy. The interpolation of the capital policy function is done by either cubic 
        interpolation or approximating the policy function with chebyshev polynomials and OLS.
        
        Returns the simulation path of capital, consumption, output and investment. Als the average and max
        Log10 euler error from the simulation
        """
        
        
        # a. initialize
        sim_k = np.zeros(self.simT+1)
        sim_k[0:2] = self.k_ss    #starting point (shock hits in second period)
        
        
        # b. interpolate 
        
        if self.sim_interpolate_type == 'chebyshev':
        
             # i. convert capital grid to chebyshev root
            roots_k = self.grid_to_nodes(self.grid_k, self.k_min, self.k_max)
            
            # ii. matrix of chebyshev polynomials
            Tk = self.chebyshev_polynomial(roots_k, self.cheb_order+1)
            
            # iii. compute OLS
            theta = la.lstsq(Tk, self.pol_kp.T, rcond=None)[0]
            
            # iv. iterate forward
            for t in range(1,self.simT):
                
                idx = self.sim_z_idx[t].astype(int)
                
                # convert capital value to chebyshev node
                root_kt = self.grid_to_nodes(sim_k[t], self.k_min, self.k_max)
                
                # matrix of chebyshev polynomials for given node
                Tkt = self.chebyshev_polynomial(root_kt, self.cheb_order+1)
                
                # interpolate 
                sim_k[t+1] = np.dot(Tkt, theta[:,idx])
                 
                
                
        if self.sim_interpolate_type == 'cubic':
        
            for t in range(1,self.simT):
                
                idx = self.sim_z_idx[t].astype(int)
                
                #cubic interpolation
                interpolant = interpolate.interp1d(self.grid_k, self.pol_kp[idx, :], kind='cubic')
                
                sim_k[t+1] = interpolant(sim_k[t])
           
            
        # d. get the other variables
        
        sim_output = self.sim_z*sim_k[0:-1]**self.alpha
        
        sim_inv = sim_k[1:] - (1-self.delta)*sim_k[0:-1]
        
        sim_cons = sim_output - sim_inv
        
        
        
        # e. euler equation error simulation
        
        euler_error_sim = np.zeros(self.simT)
        
        for t in range(self.simT):
                
            # i. initialize
            idx = self.sim_z_idx[t].astype(int)     #current shock indiex
            c = sim_cons[t]                         #current consumption
            k_plus = sim_k[t+1]                     #next period capital
            
            
            # ii. setup chebyshev approximation
            if self.sim_interpolate_type == 'chebyshev':
                
                # convert capital value to chebyshev node
                root_kp = self.grid_to_nodes(k_plus, self.k_min, self.k_max)
            
                # matrix of chebyshev polynomials for given node
                Tk_plus = self.chebyshev_polynomial(root_kp, self.cheb_order+1)
            
        
            # iii. calculate expectation on RHS of Euler equation
            avg_RHS=0
            
            for izz in range(self.Nz):  #shock in next period
                
                if self.sim_interpolate_type == 'chebyshev':
                    
                    # get c_plus in next period 
                    c_plus = self.grid_z[izz]*k_plus**self.alpha + (1-self.delta)*k_plus - np.dot(Tk_plus, theta[:,izz])
                    
                
                if self.sim_interpolate_type == 'cubic':
                    
                    # get c_plus in next period 
                    interpolant = interpolate.interp1d(self.grid_k, self.pol_kp[izz, :], kind='cubic')
                    c_plus = self.grid_z[izz]*k_plus**self.alpha + (1-self.delta)*k_plus - interpolant(k_plus)
                
                
                # RHS of Euler Equation
                A = self.u_prime(c_plus)*(self.alpha*self.grid_z[izz]*k_plus**(self.alpha-1) + (1-self.delta))
                avg_RHS += self.pi[idx,izz]*A
            
            
            #iv. calculate euler error
            euler_error_sim[t] = 1 - (self.u_prime_inv(self.beta*avg_RHS))/c
    
    
        # v. transform to log_10 and get max and average
        euler_error_sim = np.log10(np.abs(euler_error_sim))
        max_error_sim =  np.amax(euler_error_sim)
        avg_error_sim = np.mean(euler_error_sim)
                
        return sim_k, sim_cons, sim_output, sim_inv, max_error_sim, avg_error_sim
    
    
    
    ######################################
    # 4. Euler Equation Error Analysis  #
    #####################################
    
    
    
    def ee_error(self):
        """
        Computes the euler equation error over the entire state space. The function recalculates the social
        planner problem with a finer grid and then calculates the errors at each grid point.
        
        *Output
            * Log10 euler_error
            * max Log10 euler error
        """
        
                
        # a. initialize
        euler_error = np.zeros((self.Nz,self.Nk_fine))
        params_fine = self.sigma, self.beta, self.delta, self.alpha, self.grid_k_fine, self.grid_z, \
                self.Nz, self.Nk_fine, self.pi, self.maxit, self.tol
        
        
        
        # b. vfi again with finer grid
        print("\nEuler Error Calculation: Solving social planner problem on finer grid...")
        
        t0_fine = time.time() #start the clock
        
        _, pol_kp_fine, pol_cons_fine, it = vfi(params_fine)
        
        if it < self.maxit:
            print(f"\tConvergence in {self.it} iterations.")
        else : 
            print("\tNo convergence.")
            
        t1_fine = time.time()
        print(f'\tValue function iteration time elapsed: {t1_fine-t0_fine:.2f} seconds')
        
        
        
        # c. calulate euler equation error
       
        print("Euler Error Calculation: Evaluating the errors...")
        
        
        # i. approximate next period capital function with chebyshev polynomials
        if self.sim_interpolate_type == 'chebyshev':
        
            # convert capital grid to chebyshev root
            roots_k_fine = self.grid_to_nodes(self.grid_k_fine, self.k_min, self.k_max)
            
            # matrix of chebyshev polynomials
            Tk_fine = self.chebyshev_polynomial(roots_k_fine, self.cheb_order+1)
            
            # compute OLS
            theta = la.lstsq(Tk_fine, pol_kp_fine.T, rcond=None)[0]
                    
            
        
        # ii. calculate euler error at all fine grid points
        
        for iz in range(self.Nz):       #current productivity
            for ik in range(self.Nk_fine):   #current capital
            
                #initialize
                avg_RHS = 0
                c = pol_cons_fine[iz,ik]        #current consumption
                k_plus = pol_kp_fine[iz,ik]     #capital next period
                
                if self.sim_interpolate_type == 'chebyshev':
                    
                    # convert next period capital value to chebyshev node
                    root_kp_fine = self.grid_to_nodes(k_plus, self.k_min, self.k_max)
            
                    # matrix of chebyshev polynomials for given node
                    Tk_plus_fine = self.chebyshev_polynomial(root_kp_fine, self.cheb_order+1)
                
                
                
                for izz in range(self.Nz):      #next period productivity
                    
                    if self.sim_interpolate_type == 'chebyshev':
                        c_plus = self.grid_z[izz]*k_plus**self.alpha + (1-self.delta)*k_plus - np.dot(Tk_plus_fine, theta[:,izz])
                    
                    
                    if self.sim_interpolate_type == 'cubic':
                        
                        interpolant = interpolate.interp1d(self.grid_k_fine, pol_kp_fine[izz,:], kind='cubic')
                        c_plus = self.grid_z[izz]*k_plus**self.alpha + (1-self.delta)*k_plus - interpolant(k_plus)
                        
                    # RHS of Euler Equation   
                    A = self.u_prime(c_plus)*(self.alpha*self.grid_z[izz]*k_plus**(self.alpha-1) + (1-self.delta))
                    avg_RHS += self.pi[iz,izz]*A
                    
                    
                #calculate euler error
                euler_error[iz,ik] = 1 - (self.u_prime_inv(self.beta*avg_RHS))/c
        
        
        
        # iii.transform euler error with log_10 and take max
        euler_error = np.log10(np.abs(euler_error))
        max_error =  np.amax(np.amax(euler_error, axis=1))
        avg_error = np.mean(euler_error)
        
        t2_fine = time.time()
        print(f'\tError calculation time elapsed: {t2_fine-t1_fine:.2f} seconds')
        
        
        return euler_error, max_error, avg_error
            
        


    ######################
    # 5. Main Function  #
    #####################
    
    def solve_model(self):
        """
        Runs the entire model.
        """  
        
        t0 = time.time()    #start the clock
        
        
        # a. solve social planer's problem
        
        print("\nSolving social planner problem...")
    
        self.VF, self.pol_kp, self.pol_cons, self.it = vfi(self.params)
        
        if self.it < self.maxit:
            print(f"Convergence in {self.it} iterations.")
        else : 
            print("No convergence.")
            
        t1 = time.time()
        print(f'Value function iteration time elapsed: {t1-t0:.2f} seconds')
        
        
        
        # b. simulation and euler equation error
        
        print("\nMarkov Chain Simulation...")
        
        self.sim_k, self.sim_cons, self.sim_output, self.sim_inv, self.max_error_sim, self.avg_error_sim = self.simulation()

        t2 = time.time()
        print(f'Simulation time elapsed: {t2-t1:.2f} seconds')
            
        
            
        # c. calculate euler equation error
        
        if self.full_euler_error:
            self.euler_error, self.max_error, self.avg_error = self.ee_error()
            
        t3 = time.time()
        
        
        
        # d. plot
        
        if self.plott:
            
            print('\nPlotting...')
            
            # i. solutions
            plt.plot(self.grid_k, self.VF.T)
            plt.title('Value Function')
            plt.xlabel('Capital Stock')
            #plt.savefig('Figures Solution/vf_ncgm_vfi.pdf')
            plt.show()
            
            plt.plot(self.grid_k, self.pol_kp.T)
            plt.title('Next Period Capital Stock Policy Function')
            plt.xlabel('Capital Stock')
            plt.plot([self.k_min,self.k_max], [self.k_min,self.k_max],linestyle=':')
            plt.legend(['Policy Function', '45 Degree Line'])
            #plt.savefig('Figures Solution/capital_policyfun_ncgm_vfi.pdf')
            plt.show()
    
            plt.plot(self.grid_k, self.pol_cons.T)
            plt.title('Consumption Policy Function')
            plt.xlabel('Capital Stock')
            #plt.savefig('Figures Solution/consumption_policyfun_ncgm_vfi.pdf')
            plt.show()
            
            if self.full_euler_error:
                plt.plot(self.grid_k_fine, self.euler_error.T)
                plt.title('Log10 Euler Equation Error')
                plt.xlabel('Capital Stock')
                #plt.savefig('Figures Solution/log10_euler_error_ncgm_vfi.pdf')
                plt.show()
            
            # ii. simulation figures
            
            if self.simulate :
                plt.plot(np.arange(self.simT), self.sim_k[:-1])
                plt.plot(np.arange(self.simT), self.k_ss*np.ones(self.simT), linestyle='--')
                plt.title('Dynamics: Capital Stock')
                plt.xlabel('Time')
                #plt.savefig('Figures Simulation/capital_sim_ncgm_vfi.pdf')
                plt.show()
                
                plt.plot(np.arange(self.simT), self.sim_cons)
                plt.plot(np.arange(self.simT), self.c_ss*np.ones(self.simT), linestyle='--')
                plt.title('Dynamics: Consumption')
                plt.xlabel('Time')
                #plt.savefig('Figures Simulation/consumption_sim_ncgm_vfi.pdf')
                plt.show()
                
                plt.plot(np.arange(self.simT), self.sim_output)
                plt.plot(np.arange(self.simT), self.y_ss*np.ones(self.simT), linestyle='--')
                plt.title('Dynamics: Output')
                plt.xlabel('Time')
                #plt.savefig('Figures Simulation/output_sim_ncgm_vfi.pdf')
                plt.show()
                
                plt.plot(np.arange(self.simT), self.sim_inv)
                plt.plot(np.arange(self.simT), self.i_ss*np.ones(self.simT), linestyle='--')
                plt.title('Dynamics: Investment')
                plt.xlabel('Time')
                #plt.savefig('Figures Simulation/investment_sim_ncgm_vfi.pdf')
                plt.show()
                
            
            t4 = time.time()
            print(f'Plot time elapsed: {t4-t3:.2f} seconds')
            
        print("\n-----------------------------------------")
        print("Log10 Euler Equation Error Evaluation")
        print("-----------------------------------------")
        if self.full_euler_error:
            print(f"\nFull Grid Evalulation: Max Error  = {self.max_error:.2f}")
            print(f"Full Grid Evalulation: Max Error: Average Error = {self.avg_error:.2f}")
        
        if self.simulate:
            print(f"\nSmiluation: Max Error  = {self.max_error_sim:.2f}")
            print(f"Simulation: Average Error = {self.avg_error_sim:.2f}")
            
        t5 = time.time()
        print(f'\nTotal Run Time: {t5-t0:.2f} seconds')
  




#########################
# II. Jitted Functions  #
########################

#########################
# 1. Helper Functions  #
########################
        

@njit    
def utility(c, sigma):
    eps = 1e-8
    
    if  sigma == 1:
        return np.log(np.fmax(c, eps))
    else:
        return (np.fmax(c, eps) ** (1 - sigma) -1) / (1 - sigma)
    
    
    
    
#################################
# 2. Value Function Iteration  #
################################

@njit(parallel=True)
def vfi(params):
   
    """
    Value function iteration to solve the social planner problem.
    
    *Input
        * Parameters of the model
    
    *Output
        * VF is value function
        * pol_kp is k' or savings policy function
        * pol_cons is the consumption policy function
        * it is the iteration number 
    """

    # a. Initialize counters, initial guess. storage matriecs
    
    sigma, beta, delta, alpha, grid_k, grid_z, Nz, Nk, pi, maxit, tol = params
    
    VF_old  = np.zeros((Nz,Nk))  #initial guess
    VF = np.copy(VF_old)    #contracted value function aka Tv
    pol_kp = np.copy(VF_old)    # next period capital (k') policy function
    pol_cons = np.copy(VF_old)  # consumption policy funnction
    
    # b. Iterate
    for it in range(maxit) :
        for iz in range(Nz):
            for ik in prange(Nk):
                
                # i. resource constrant
                c = grid_z[iz]*grid_k[ik]**alpha + (1-delta)*grid_k[ik] - grid_k
                
                # ii. utility and impose nonnegativity for consumption
                util = utility(c, sigma)
                util[c < 0] = -10e10
                
                # iii. value and policy functions
                RHS = util + beta*np.dot(pi[iz,:],VF_old)   #RHS of Bellman
                
                VF[iz,ik] = np.max(RHS) #take maximum value for value function
                
                pol_kp[iz,ik] = grid_k[np.argmax(RHS)]    #policy function for how much to save
                
            # consumption policy function
            pol_cons[iz,:] = grid_z[iz]*grid_k**alpha + (1-delta)*grid_k - pol_kp[iz,:]
       
        # iv. calculate distance from previous iteration
        dist = la.norm(VF-VF_old)
       
        if dist < tol :
            break
        
        VF_old = np.copy(VF)
 
    
    return VF, pol_kp, pol_cons, it
        


# run everything
ncgm_stoch = ncgmVFI()
ncgm_stoch.solve_model()

