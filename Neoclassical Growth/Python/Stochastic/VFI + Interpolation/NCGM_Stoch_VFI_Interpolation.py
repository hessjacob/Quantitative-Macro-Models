
"""
Author: Jacob Hess
Date: July 2021

Description: This code solves the social planner's problem for the stochastic neoclassical growth model. Since the 
welfare theorems hold the solution coincides with the competitive equilibrium solution. The model is solved using 
value function interation and interpolates with cubic splines to get the value for the value function 
between the grid points. The code is a loose adaption from Fabrice Collard's notes. The computation requires less 
iterations and is significantly faster than conventional value function iteration. After obtaining the solution
the code simulates a markov chain where it interpolates the next period capital policy function by either cubic 
interpolation or by approximating it with chebyshev polynomials and OLS. I also calculate the euler equation errors 
to assess the accuracy of the solution two different ways. One is by calculating the error for each period in the 
simulation and the other is by calculating the error over the entire state space.

Acknowledgements:
    1) Heer and Maussner Ch. 4
    2) Fabrice Collard's "Value Iteration" notes (http://fabcol.free.fr/notes.html)
    3) Wouter den Haan's "accuracy tests" notes (https://www.wouterdenhaan.com/notes.htm)
    
Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual
    -- QuantEcon (to install: 'conda install quantecon')

Requirements file:
    -- Accompanying requirements.txt contains the versions of the library and packages versions that I used.
    -- Not required to use, but I recommend doing so if you either have trouble running this file or figures generated do not coincide with mine. 
    -- In your termain run the following 
        * pip install -r /your path/requirements.txt
"""


import time
import numpy as np
from numpy import linalg as la
from scipy import interpolate
import quantecon as qe
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')



#############
# I. Model  #
############


class ncgmVFIandINTERPOLATE:
    
    ############
    # 1. setup #
    ############

    def __init__(self, plott = 1, simulate=1, sim_interpolate_type = 'chebyshev', full_euler_error = 0):              
            
            #parameters subject to changes
            self.plott = plott  #select 1 to make plots
            self.simulate = simulate  #select 1 to run simulation
            self.sim_interpolate_type = sim_interpolate_type  #interpolation during simulation. options: 'chebyshev, 'cubic'. if simulate=0 this option is automatically ignored.
            self.full_euler_error = full_euler_error  #select 1 to calculate euler error on entire grid
            
            self.setup_parameters()
            self.setup_grid()
            self.setup_markov()
            
            # warnings
            if self.plott != 1 and self.plott != 0:
                raise Exception("Plot option incorrectly entered: Choose either 1 or 0.")
            
            if self.simulate != 1 and self.simulate != 0:
                raise Exception("Simulate option incorrectly entered: Choose either 1 or 0.")
                
            if self.sim_interpolate_type != 'chebyshev' and self.sim_interpolate_type != 'cubic':
                raise Exception("Interpolation for simulation option incorrectly entered: Choose either 'chebyshev' or 'cubic'")
                
            if self.full_euler_error != 1 and self.full_euler_error != 0:
                raise Exception("Euler error full grid evaluation option incorrectly entered: Choose either 1 or 0.")
            
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
        self.Nk = 30
        self.dev = 0.9
        self.k_min = (1-self.dev)*self.k_ss
        self.k_max = (1+self.dev)*self.k_ss
        
        # consumption grid
        self.Nc = 500
        self.c_min = 0.01
        self.c_max = self.k_max**self.alpha
        
        # e. simulation
        self.seed = 123
        self.cheb_order = 10    #chebyshev polynomial order
        self.simT = 200    #number of simulation periods
            
        # f. euler equation error analysis
        if self.full_euler_error:
            self.Nk_fine=100
            self.Nc_fine = 1500
            
        
        
    def setup_grid(self):
        # a. capital grid
        self.grid_k = np.linspace(self.k_min, self.k_max, self.Nk)

        # b. consumption grid
        self.grid_c = np.linspace(self.c_min, self.c_max, self.Nc)
        
        # c. finer grids for euler error analysis
        if self.full_euler_error:
            self.grid_k_fine = np.linspace(self.k_min, self.k_max, self.Nk_fine)
            self.grid_c_fine = np.linspace(self.c_min, self.c_max, self.Nc_fine)



    def setup_markov(self):
        
        # a. discretely approximate the continuous income process 
        self.mc = qe.markov.approximation.rouwenhorst(n=self.Nz, rho=self.rho_z, sigma=self.sigma_z, mu=self.z_bar)
        #self.mc = qe.markov.approximation.tauchen(n=self.Nz, rho=self.rho_z, sigma=self.sigma_z, mu=self.z_bar, n_std=3)

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
    
    
    
    def utility(self, c):
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
    
    
    def UnivariateSpline_multi(self, x, y, x_i):
        
        ncol = y.shape[0] 
        
        y_i = np.zeros((ncol,len(x_i)))
        
        for i in range(ncol):
            interpolant = interpolate.UnivariateSpline(x, y[i,:], k=3, s=0) 
            #k is degree of spline. 1 means linear, 3 cubic.. s indicates that...
            #...the given data are the knots and that the spline function passes through all of them
            y_i[i,:] = interpolant(x_i)
    
        return y_i


    
    #################################
    # 3. Value Function Iteration  #
    ################################
    
    def vfi(self, grid_k, Nk):
       
        """
        Value function iteration to solve the social planner problem.
        
        *Input 
            * grid_k is the capital grid
            * Nk is the number of grid points
        
        *Output
            * VF is value function
            * pol_kp is k' or savings policy function
            * pol_cons is the consumption policy function
            * it is the iteration number 
        """
    
        # a. Initialize counters, initial guess. storage matriecs
        
        VF_old = np.linspace(0,1,Nk)*np.ones((self.Nz, Nk))  #initial guess
        VF = np.copy(VF_old)    #contracted value function aka Tv
        pol_kp = np.copy(VF_old)    # next period capital (k') policy function
        pol_cons = np.copy(VF_old)      #consumption policy function
        
        util = self.utility(self.grid_c)   #utility evaluated on the consumption grid
        
        # b. Iterate
        for it in range(self.maxit) :
            for iz in range(self.Nz):
                for ik in range(Nk):
                    
                    # i. resource constrant
                    kp = self.grid_z[iz]*grid_k[ik]**self.alpha + (1-self.delta)*grid_k[ik] - self.grid_c
                    
                    # ii. interpolate 
                    
                    VF_int = self.UnivariateSpline_multi(grid_k, VF_old, kp)
                    
                    # iii. value and policy functions
                    RHS = util + self.beta*np.dot(self.pi[iz,:],VF_int)   #RHS of Bellman
                    
                    VF[iz,ik] = np.max(RHS) #take maximum value for value function
                    
                    # iv. consumption policy function 
                    pol_cons[iz,ik] = self.grid_c[np.argmax(RHS)]
                
                # vi. next period capital policy function
                pol_kp[iz,:] = self.grid_z[iz]*grid_k**self.alpha +(1-self.delta)*grid_k - pol_cons[iz,:]
            
            # v. calculate supnorm
            dist = np.abs(VF-VF_old).max()
           
            if dist < self.tol :
                break
            
            VF_old = np.copy(VF)
     
        
        return VF, pol_kp, pol_cons, it



    ##################
    # 4. Simulation #
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
                    c_plus = self.grid_z[izz]*sim_k[t+1]**self.alpha + (1-self.delta)*sim_k[t+1] - np.dot(Tk_plus, theta[:,izz])
                    
                
                if self.sim_interpolate_type == 'cubic':
                    
                    # get c_plus in next period 
                    interpolant = interpolate.interp1d(self.grid_k, self.pol_kp[izz, :], kind='cubic')
                    c_plus = self.grid_z[izz]*sim_k[t+1]**self.alpha + (1-self.delta)*sim_k[t+1] - interpolant(k_plus)
                
                # RHS of Euler Equation
                A = self.u_prime(c_plus)*(self.alpha*self.grid_z[izz]*k_plus**(self.alpha-1) + (1-self.delta))
                avg_RHS += self.pi[idx,izz]*A
            
            
            #iv. calculate euler error
            euler_error_sim[t] = 1 - self.u_prime_inv(self.beta*avg_RHS)/c
    
    
        # v. transform to log_10 and get max and average
        euler_error_sim = np.log10(np.abs(euler_error_sim))
        max_error_sim =  np.amax(euler_error_sim)
        avg_error_sim = np.mean(euler_error_sim)
                
        
            
        return sim_k, sim_cons, sim_output, sim_inv, max_error_sim, avg_error_sim
    
    
    
    ######################################
    # 5. Euler Equation Error Analysis  #
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
        
    
        # b. vfi again with finer grid
        print("\nEuler Error Calculation: Solving social planner problem on finer grid...")
        
        t0_fine = time.time() #start the clock
        
        _, pol_kp_fine, pol_cons_fine, it = self.vfi(self.grid_k_fine, self.Nk_fine)
        
        if it < self.maxit-1:
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
                    
            
        
        # ii. calculate euler error at all grid points
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
                euler_error[iz,ik] = 1 - self.u_prime_inv(self.beta*avg_RHS)/c
        
        
        
        # iii.transform euler error with log_10 and take max
        euler_error = np.log10(np.abs(euler_error))
        max_error =  np.amax(np.amax(euler_error, axis=1))
        avg_error = np.mean(euler_error)
        
        t2_fine = time.time()
        print(f'\tError calculation time elapsed: {t2_fine-t1_fine:.2f} seconds')
        
        
        return euler_error, max_error, avg_error




    ######################
    # 6. Main Function  #
    #####################
    
    def solve_model(self):
        """
        Runs the entire model.
        """  
        
        t0 = time.time()    #start the clock
        
        
        # a. solve social planer's problem
        
        print("\nSolving social planner problem...")
    
        self.VF, self.pol_kp, self.pol_cons, self.it = self.vfi(self.grid_k, self.Nk)
        
        if self.it < self.maxit-1:
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
            #plt.savefig('Figures Solution/vf_ncgm_vfi_interpolate.pdf')
            plt.show()
            
            plt.plot(self.grid_k, self.pol_kp.T)
            plt.title('Next Period Capital Stock Policy Function')
            plt.xlabel('Capital Stock')
            plt.plot([self.k_min,self.k_max], [self.k_min,self.k_max],linestyle=':')
            plt.legend(['Policy Function', '45 Degree Line'])
            #plt.savefig('Figures Solution/capital_policyfun_ncgm_vfi_interpolate.pdf')
            plt.show()
    
            plt.plot(self.grid_k, self.pol_cons.T)
            plt.title('Consumption Policy Function')
            plt.xlabel('Capital Stock')
            #plt.savefig('Figures Solution/consumption_policyfun_ncgm_vfi_interpolate.pdf')
            plt.show()
            
            if self.full_euler_error:
                plt.plot(self.grid_k_fine, self.euler_error.T)
                plt.title('Log10 Euler Equation Error')
                plt.xlabel('Capital Stock')
                #plt.savefig('Figures Solution/log10_euler_error_ncgm_vfi_interpolate.pdf')
                plt.show()
            
            # ii. simulation figures
            
            if self.simulate :
                plt.plot(np.arange(self.simT), self.sim_k[:-1])
                plt.plot(np.arange(self.simT), self.k_ss*np.ones(self.simT), linestyle='--')
                plt.title('Dynamics: Capital Stock')
                plt.xlabel('Time')
                #plt.savefig('Figures Simulation/capital_sim_ncgm_vfi_interpolate.pdf')
                plt.show()
                
                plt.plot(np.arange(self.simT), self.sim_cons)
                plt.plot(np.arange(self.simT), self.c_ss*np.ones(self.simT), linestyle='--')
                plt.title('Dynamics: Consumption')
                plt.xlabel('Time')
                #plt.savefig('Figures Simulation/consumption_sim_ncgm_vfi_interpolate.pdf')
                plt.show()
                
                plt.plot(np.arange(self.simT), self.sim_output)
                plt.plot(np.arange(self.simT), self.y_ss*np.ones(self.simT), linestyle='--')
                plt.title('Dynamics: Output')
                plt.xlabel('Time')
                #plt.savefig('Figures Simulation/output_sim_ncgm_vfi_interpolate.pdf')
                plt.show()
                
                plt.plot(np.arange(self.simT), self.sim_inv)
                plt.plot(np.arange(self.simT), self.i_ss*np.ones(self.simT), linestyle='--')
                plt.title('Dynamics: Investment')
                plt.xlabel('Time')
                #plt.savefig('Figures Simulation/investment_sim_ncgm_vfi_interpolate.pdf')
                plt.show()
                
            
            t4 = time.time()
            print(f'Plot time elapsed: {t4-t3:.2f} seconds')
            
        print("\n-----------------------------------------")
        print("Log10 Euler Equation Error Evaluation")
        print("-----------------------------------------")
        if self.full_euler_error:
            print(f"\nFull Grid Evalulation: Max Error  = {self.max_error:.2f}")
            print(f"Full Grid Evalulation: Average Error = {self.avg_error:.2f}")
        
        if self.simulate:
            print(f"\nSmiluation: Max Error  = {self.max_error_sim:.2f}")
            print(f"Simulation: Average Error = {self.avg_error_sim:.2f}")
            
        t5 = time.time()
        print(f'\nTotal Run Time: {t5-t0:.2f} seconds')
  
    
#run everything
ncgm_det =ncgmVFIandINTERPOLATE()
ncgm_det.solve_model()

