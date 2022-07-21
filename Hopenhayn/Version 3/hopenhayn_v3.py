
"""
Author: Jacob Hess

First Version: May 2021
This Version: July 2022

Description: This is a firm dynamics model from Hopenhayn (1992) augmented to include capital accumulation and adjustment costs. Firms own their own capital 
stock and make investment decision subject to a convex and non-convex costs. The code solves for the stationary equilibrium where entry/exit arises endogenously
from the solution. The economy is in partial equilibrium where frictionless labor and capital markets clear, but I do not specify a goods market. I normalize
aggregate labor supply to 1 and the wage is endogenously determined such that the free entry condition is satisfied. 

Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual)
    -- QuantEcon (to install: 'conda install quantecon')
    -- Interpolation from EconForge
       * optimized interpolation routines for python/numba
       * to install 'conda install -c conda-forge interpolation'
       * https://github.com/EconForge/interpolation.py
"""

import time
import numpy as np
from scipy import stats
from numba import njit, prange
import quantecon as qe
from interpolation import interp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')




##############
# I. Model  #
#############

class HopenhaynV3:
    
    """
    Class object of the model. HopenhaynV3().solve_model() runs everything
    """

    ############
    # 1. setup #
    ############

    def __init__(self, plott =1):             #select 1 to make plots
    
        #parameters subject to changes
        self.plott = plott
        
        self.setup_parameters()
        self.setup_grid()
        self.setup_discretization()
        
        #pack parameters for jitted functions
        self.params_vfi = self.alpha, self.beta, self.delta, self.gamma, self.cf, self.psi, self.xi, self.pi, self.grid_k, self.grid_z, self.maxit, self.tol
        self.params_dist = self.grid_k, self.Nz, self.pi, self.pi_stat, self.nu, self.maxit, self.tol
    
    def setup_parameters(self):
        
        # a. model parameters
        self.beta = 0.9615          #annual discount factor 
        self.alpha = 0.85/3         #capital share
        self.gamma = 0.85*2/3       #labor share
        self.delta = 0.1            #annual depreciation rate
        self.psi = 0.5            #capital adjustment parameter
        self.xx = 1 - self.alpha - self.gamma   #for factor demand solution
        self.cf = 0.5               #fixed cost
        self.ce = 0.5                 #entry cost
        self.psi = 0.25             #convex adjustment cost parameter
        self.xi = 0.01              #non-convex/fixed adjustment cost
        self.ck = 1               #price of capital at entry
        self.interest_rate = 1/self.beta - 1    #steady state interest rate
        self.b = 2                  #shape parameter for pareto initial productivity distribution
        
        # AR(1) productivity process
        self.rho_z = 0.6            #autocorrelation coefficient
        self.sigma_z = 0.2          #std. dev. of shocks
        self.Nz = 10                #number of discrete income states
        self.z_bar = 0              #constant term in continuous productivity process (not the mean of the process)
        
        # b. iteration parameters  
        self.tol = 1e-8                         #default tolerance
        self.tol_w = 1e-4                       #tolerance for wage bisection
        self.maxit = 2000                       #maximum iterations
        
        # c. capital grid
        self.Nk = 500       # number of capital grid points
        self.k_min = 0.01   # minimum capital level
        self.k_max = 40    # maximum capital level
        self.curv = 3       # grid curvature parameter
        
    def setup_grid(self) :
        
        # a. capital grid
        self.grid_k = self.make_grid(self.k_min, self.k_max, self.Nk, self.curv)
        
    def setup_discretization(self):
        
        # a. discretely approximate the continuous tfp process 
        self.mc = qe.markov.approximation.rouwenhorst(self.Nz, self.z_bar, self.sigma_z, self.rho_z)
        #self.mc = qe.markov.approximation.tauchen(self.rho_z, self.sigma_z, self.z_bar, 3, self.Nz)

        # b. transition matrix and states
        self.pi = self.mc.P
        self.pi_stat = self.mc.stationary_distributions.T.ravel()   #ravel to resize array from 2d to 1d for rv_discrete
        self.grid_z = np.exp(self.mc.state_values)
        
        # c. initial productivity distribution for entrant firm
        
        #Pareto distribution with parameter 2. The distribution is shifted -(1-grid_z[0]) to the left so that it starts a z_min.
        #The shift is identical to as if you add (1-grid_z[0])) to grid_z making z_min=1. I normalize the mass to 1.
        
        self.nu = stats.pareto.pdf(self.grid_z, self.b,loc=-(1-self.grid_z[0]))/np.sum(stats.pareto.pdf(self.grid_z, self.b,loc=-(1-self.grid_z[0])))
        
        
        
        
    ####################################
    # 2. Helper functions #
    ####################################
    
    
    
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
    

    
    
    
    
    ################################################
    # 3. Solve incumbent and entrant firm problem #
    ###############################################
    
    
    def entrant_firm(self, VF):
        """
        Entrant firm chooses its initial investment plus expected firm value over the initial productivity distribution.
        """
    
        RHS_entrant = np.zeros(self.Nk)
        RHS_entrant = - self.ck*self.grid_k + self.beta * np.dot(self.nu, VF) 
    
        VF_entrant = np.max(RHS_entrant)    #value of the entrant
        k_e = self.grid_k[np.argmax(RHS_entrant)] #optimal initial investment
        
        return VF_entrant, k_e
    
    
    
    def find_equilibrium_wage(self):
        """
        Using the bisection method this function finds the unique equilibrium wage that clears the labor market and satisfies the free entry condition. 
        
        In a stationary equilibrium the free entry condition (or the present discounted value of the entrant) is zero. The free entry condition is where 
        the value of the entrant equals zero (more precisely, where the expected firm value over the initial productivity distribution plus its initial 
        investment equals the cost of entry).
        """
        
        # a. set up the wage interval
        wmin, wmax = 0.01, 100
        
        for it_w in range(self.maxit):
            
            print("\n-----------------------------------------")
            print("Iteration #"+str(it_w+1))
            
            # i. guess a price
            
            if it_w == 0 :
                wage_guess =0.5
            else:
                wage_guess = (wmin+wmax)/2
            
            
            # ii. incumbent firm value function. present discounted value of incumbant
            print("Solving incumbent firm problem...")
            
            t1 = time.time()
            
            VF, _, _, _, _, _, _, it_vfi  = incumbent_firm(wage_guess, self.params_vfi)
            
            if it_w < self.maxit-1:
                print(f"Value function convergence in {it_vfi} iterations.")
            else : 
                raise Exception("No value function convergence.")
                
            t2 = time.time()
            print(f'Incumbent firm time elapsed: {t2-t1:.2f} seconds')

            # iii. free entry condition
            
            print("\nSolving entrant firm problem...")
            
            t3=time.time()
            
            VF_entrant = self.entrant_firm(VF)[0]   #value of the entrant
            
            t4=time.time()
            print(f'Entrant firm time elapsed: {t4-t3:.2f} seconds')
            
            
            #iv. calculate free entry condition and check if satisfied
            free_entry_cond = VF_entrant - self.ce
            
            if np.abs(free_entry_cond) < self.tol_w:
                print("\n-----------------------------------------")
                print('\nConvergence!')
                
                wage_ss = wage_guess
                break
            
            # v. update price interval
            else:
                if free_entry_cond < 0 :
                    wmax=wage_guess 
                else:
                    wmin=wage_guess
                    
            print(f"\nNew wage guess = {wage_guess:.5f} \t Free entry condition = {free_entry_cond:8.5f}")
            
        if it_w > self.maxit-1 :
            print("No convergence")
            
        
        return wage_ss





    #####################
    # 3. Main function #
    ####################
    
    def solve_model(self):
        """
        Finds the stationary equilibrium.
        """  
        
        t0 = time.time()    #start the clock
        
        
        # a. Find the steady state wage using bisection
        print('\nFinding wage that satisifies free entry condition...')
        
        self.wage_ss = self.find_equilibrium_wage()
        
        
        # b. Use the equilibrium wage to recover incumbent and entrant firm solutions
        print("\nRecovering equilibrium solutions...")
        self.VF, self.pol_kp, self.pol_n, self.pol_continue, self.pol_inv, self.firm_output, self.firm_profit, self.it_vf = incumbent_firm(self.wage_ss, self.params_vfi)     
        
        self.VF_entrant, self.k_e = self.entrant_firm(self.VF)
        
        t5 = time.time()
        print('\nFree entry condition bisection time elapsed: '+str(time.strftime("%M:%S",time.gmtime(t5-t0))))
        
        
        # c. Invariant joint distribution with endogenous exit
        print('\nFinding stationary density function by forward iteration...')
        
        self.stationary_pdf_hat, self.it_pdf, self.dist_pdf = discrete_stationary_density(self.pol_kp, self.k_e, self.pol_continue, self.params_dist)
        
        if self.it_pdf < self.maxit-1:
            print(f"Convergence in {self.it_pdf} iterations.")
        else : 
            print(f"Maximum iteration number reached. Distance between last iteration: {self.dist_pdf:8.5f}")
        
        
        # d. Mass of entrants (m_star) in the ss equilibrium. Because labor is supplied inelastically we can use the labor market clearing condition
        # to solve m_star
        self.m_star = 1/np.sum(np.sum(self.stationary_pdf_hat*self.pol_n))
        
        
        # e. Rescale to get invariant joint distribution (mass of plants)
        self.stationary_pdf = self.m_star * self.stationary_pdf_hat
        
        # stationary distribution by percent 
        self.stationary_pdf_star = self.stationary_pdf/np.sum(np.sum(self.stationary_pdf))
        
        t6 = time.time()
        print(f'Density approximation time elapsed: {t6-t5:.2f} seconds')
        
        
        # f. marginal distributions
        
        print("\nCalculating aggergate statistics and marginal densities...")
        
        # marginal capital density by percent
        self.capital_marginal_pdf_star = np.sum(self.stationary_pdf_star, axis=0) 
        
        # employment (capital) density 
        self.emp_pdf = (self.pol_n * self.stationary_pdf)
        
        # marginal (capital) employment density by percent
        self.emp_marginal_pdf = np.sum(self.emp_pdf, axis=0) / np.sum(np.sum(self.emp_pdf,axis=0))
        self.emp_marginal_cdf = np.cumsum(self.emp_marginal_pdf)
        
        
        # g. aggregate statistics
        
        self.Y_ss = np.sum(np.sum(self.firm_output*self.stationary_pdf))
        self.K_ss = np.sum(np.sum(self.stationary_pdf*self.grid_k))
        self.N_ss = np.sum(np.sum(self.stationary_pdf*self.pol_n)) #should equal 1
        self.Inv_ss = np.sum(np.sum(self.stationary_pdf*self.pol_inv))
        self.TFP_ss = self.Y_ss/(self.K_ss**self.alpha * self.N_ss**self.gamma)
        
        self.average_incumbent_firm_size = np.sum(np.sum(self.stationary_pdf_star*self.pol_n))  
        #self.average_firm_size = self.N_ss / np.sum(np.sum(self.stationary_pdf)) alternative calculation
        self.average_entrant_firm_size = np.sum(self.nu*self.pol_n[:,(np.sum(self.grid_k <= self.k_e) - 1)]) 
        
        self.exit_rate = 1 - np.sum(np.dot(self.pi.T, self.stationary_pdf_hat)*self.pol_continue)/np.sum(self.stationary_pdf_hat) 
        
        t7 = time.time()
        print(f'Calculation time: {t7-t6:.2f} seconds')
        
        # h. plot
        
        if self.plott:
            print("\nPlotting...")
            
            idx = [0, 2, 4, 6, 9]
            for ii in idx :
                plt.plot(self.grid_k, self.VF[ii,:])
            plt.legend(['V(k,z$_{'+str(idx[0])+'}$)','V(k,z$_{'+str(idx[1])+'}$)', 'V(k,z$_{'+str(idx[2])+'}$)','V(k,z$_{'+str(idx[3])+'}$)','V(k,z$_{'+str(idx[4])+'}$)'])
            plt.title('Incumbant Firm Value Function')
            plt.xlabel('Capital')
            #plt.savefig('vf_hopehaynv3.pdf')
            plt.show()
            
            for ii in idx :
                plt.plot(self.grid_k, self.pol_kp[ii,:])
            plt.plot(self.grid_k, (1-self.delta)*self.grid_k,':')
            plt.legend(["k'(k,z$_{"+str(idx[0])+"}$)","k'(k,z$_{"+str(idx[1])+"}$)", "k'(k,z$_{"+str(idx[2])+"}$)","k'(k,z$_{"+str(idx[3])+"}$)","k'(k,z$_{"+str(idx[4])+"}$)", "k(1-$\delta$)"])
            plt.title('Capital Next Period Policy Funciton')
            plt.xlabel('Capital')
            #plt.savefig('pol_k_hopehaynv3.pdf')
            plt.show()
            
            for ii in idx :
                plt.plot(self.grid_k, self.pol_inv[ii,:])
            plt.legend(['i(k,z$_{'+str(idx[0])+'}$)','i(k,z$_{'+str(idx[1])+'}$)', 'i(k,z$_{'+str(idx[2])+'}$)', 'i(k,z$_{'+str(idx[3])+'}$)','i(k,z$_{'+str(idx[4])+'}$)'])
            plt.title("Investment: $k'(k,z)-(1-\delta)k$")
            plt.xlabel('Capital')
            #plt.savefig('pol_inv_hopehaynv3.pdf')
            plt.show()
            
            for ii in idx :
                plt.plot(self.grid_k, self.pol_n[ii,:])
            plt.legend(['n(k,z$_{'+str(idx[0])+'}$)','n(k,z$_{'+str(idx[1])+'}$)', 'n(k,z$_{'+str(idx[2])+'}$)', 'n(k,z$_{'+str(idx[3])+'}$)', 'n(k,z$_{'+str(idx[4])+'}$)'])
            plt.title("Labor Demand Policy Function")
            plt.xlabel('Capital')
            #plt.savefig('pol_n_hopehaynv3.pdf')
            plt.show()
            
            plt.plot(self.grid_k, self.capital_marginal_pdf_star)
            plt.plot(self.grid_k, self.emp_marginal_pdf)
            plt.title('Stationary Marginal Densities' )
            plt.xlabel('Capital')
            plt.ylabel('Percent')
            plt.legend(['Productivity','Employment'])
            #plt.savefig('pdf_hopehaynv3.pdf')
            plt.show()
            
            t8 = time.time()
            print(f'Plot time elapsed: {t8-t7:.2f} seconds')
        
        print("\n-----------------------------------------")
        print("Stationary Equilibrium")
        print("-----------------------------------------")
        print(f"ss wage  = {self.wage_ss:.2f}")
        print(f"exit rate = {self.exit_rate:.3f}")
        print(f"avg. incumbent firm size = {self.average_incumbent_firm_size:.2f}")
        print(f"avg. entrant firm size = {self.average_entrant_firm_size:.2f}")
        print(f"\nss output = {self.Y_ss:.2f}")
        print(f"ss investment = {self.Inv_ss:.2f}")
        print(f"ss tfp = {self.TFP_ss:.2f}")
        print(f"ss capital = {self.K_ss:.2f}")
        print(f"ss labor = {self.N_ss:.2f}")
        
        t9 = time.time()
        print('\nTotal Run Time: '+str(time.strftime("%M:%S",time.gmtime(t9-t0))))
        
        
        
        
        
#########################
# II. Jitted Functions #
########################

###########################################################
# 1. Incumbent firm problem and Value Function Iteration #
##########################################################


@njit(parallel=True)
def incumbent_firm(wage, params_vfi):
    """
    Value function iteration for the incumbent firm problem.
    
    *Input
        - wage
        - params_vfi: model parameters
        
    *Output VF, pol_kp, pol_n, pol_continue, pol_inv, firm_output, firm_profit, it   
        - VF: Incumbent firm value function
        - pol_kp: k prime aka firm's capital stock next period
        - pol_n: labor demand
        - pol_continue: policy function where when 1 firm continues and when 0 firm exits the market
        - pol_inv: investment policy function, k'-(1-delta)k
        - firm_output: production after plugging in pol_n for labor input
        - firm_profit: per-period profits
        - it: number of iterations
    """ 

    
    #a. Initialize counters, initial guess, storage matrices
    
    alpha, beta, delta, gamma, cf, psi, xi, pi, grid_k, grid_z, maxit, tol = params_vfi
    
    Nz = len(grid_z)
    Nk = len(grid_k)
    
    VF_old    = np.zeros((Nz,Nk))  #initial guess
    VF = np.copy(VF_old)
    pol_kp = np.copy(VF_old)
    pol_n = np.copy(VF_old)
    pol_inv = np.copy(VF_old)
    pol_continue = np.copy(VF_old)
    firm_output = np.copy(VF_old)
    firm_profit = np.copy(VF_old)
    
    VF_invest = np.zeros((Nz,Nk))
    VF_inaction = np.zeros((Nz,Nk))
    VF_exit = np.zeros((Nz,Nk))
    
    
    # b. iterate
    for it in range(maxit):
        for iz in prange(Nz):
            for ik in range(Nk):
                
                pol_n[iz,ik] = ((grid_z[iz] * gamma) / wage)**(1/(1-gamma)) * grid_k[ik]**(alpha/(1-gamma))
                    
                # i. solution to static problem
                
                firm_output[iz,ik] = grid_z[iz] * grid_k[ik] ** alpha * pol_n[iz,ik] ** gamma 
                
                firm_profit[iz,ik] = firm_output[iz,ik] - wage * pol_n[iz,ik] 
                    
                
                # ii. continuation values
                
                # (1) the value of investing
                RHS_invest = -grid_k + (1-delta)*grid_k[ik] - (psi/2)*((grid_k-(1-delta)*grid_k[ik]) / grid_k[ik])**2 * grid_k[ik] \
                   - xi*grid_k[ik] + beta*np.dot(pi[iz,:], VF_old)
                
                VF_invest[iz,ik] = np.max(RHS_invest)
                
                # (2) value of waiting. k'=(1-delta)k
                
                RHS_inaction=0
                
                # calculate expected value
                for izz in range(Nz):
                    
                    RHS_inaction += pi[iz,izz]*interp(grid_k,VF_old[izz,:],(1-delta)*grid_k[ik])
                
                VF_inaction[iz,ik] = beta*RHS_inaction
                
                # (3) value of exitting. x=-(1-delta)k, hence k'=0
                VF_exit[iz,ik] = (1-delta) * grid_k[ik] - (psi/2)* (delta-1)**2 * grid_k[ik] - xi*grid_k[ik]     
                
                
                # iii. value of the incumbent firm
                
                vf_array = np.array([VF_invest[iz,ik] - cf, VF_inaction[iz,ik] - cf, VF_exit[iz,ik]])
                VF[iz,ik] = firm_profit[iz,ik] + np.max(vf_array)
                
                if (VF_invest[iz,ik] - cf) == np.max(vf_array): #invest
                    pol_kp[iz,ik] = grid_k[np.argmax(RHS_invest)]
                    pol_inv[iz,ik] = pol_kp[iz,ik] - (1-delta) * grid_k[ik]
                    pol_continue[iz,ik] = 1
                
                elif (VF_inaction[iz,ik] - cf) == np.max(vf_array): #inaction
                    pol_kp[iz,ik] = (1-delta)*grid_k[ik]
                    pol_inv[iz,ik] = 0
                    pol_continue[iz,ik] = 1
                    
                else: #exit
                    pol_kp[iz,ik] = 0
                    pol_inv[iz,ik] = -(1-delta) * grid_k[ik]
                    pol_continue[iz,ik] = 0
                    
                
        
        # iv. calculate supremum norm
        dist = np.abs(VF - VF_old).max()
    
        if dist < tol :
           break
       
        VF_old = np.copy(VF)


    return VF, pol_kp, pol_n, pol_continue, pol_inv, firm_output, firm_profit, it   





####################################
# 2. Find stationary distribution #
###################################

@njit
def discrete_stationary_density(pol_kp, k_e, pol_continue, params_dist):
    """
    Discrete approximation of the density function. Approximates the stationary joint density through forward 
    iteration over a discretized state space. The algorithm is from Ch.7 in Heer and Maussner.
    
    *Input
        - pol_kp: k prime aka firm's capital stock next period
        - k_e: optimal level of capital for entrant firm
        - pol_continue: policy function where when 1 firm continues and when 0 firm exits the market
        - params_dist: model parameters
        
    *Output
        - stationary_pdf: joint stationary density function
        - it: number of iterations
        - dist: supremum norm between stationary_pdf and previous iteration
    """
    
    # a. initialize
    
    grid_k, Nz, pi, pi_stat, nu, maxit, tol = params_dist
    
    Nk = len(grid_k)
    m = 1   #we normalize the mass of potential entrants to one
    
    # initial guess uniform distribution
    stationary_pdf_old = np.ones((Nk, Nz))/Nk
    stationary_pdf_old = stationary_pdf_old * np.transpose(pi_stat)
    stationary_pdf_old = stationary_pdf_old.T
    
    # b. fixed point iteration
    for it in range(maxit):   # iteration 
        
        stationary_pdf = np.zeros((Nz, Nk))    # distribution in period t+1
             
        for iz in range(Nz):     # iteration over productivity types in period t
            
            for ia, a0 in enumerate(grid_k):  # iteration over assets in period t   
            
                # i. get k'
                k_prime = pol_kp[iz,ia]
                
                # ii. obtain distribution in period t+1   
                
                #left edge of the grid
                if k_prime <= grid_k[0]:
                    for izz in range(Nz):
                        stationary_pdf[izz,0] = stationary_pdf[izz,0] + stationary_pdf_old[iz,ia] * pi[iz,izz] * pol_continue[iz, ia] 
                        
                
                #right edge of the grid
                elif k_prime >= grid_k[-1]:
                    for izz in range(Nz):
                        stationary_pdf[izz,-1] = stationary_pdf[izz,-1] + stationary_pdf_old[iz,ia] * pi[iz,izz] * pol_continue[iz, ia] 
                        
                    
                #inside the grid range
                else:
                    
                    j = np.sum(grid_k <= k_prime) - 1   #grid index where k' is located
                    
                    for izz in range(Nz):
                        stationary_pdf[izz,j] =stationary_pdf[izz,j] + (stationary_pdf_old[iz,ia]*pi[iz,izz]*pol_continue[iz, ia] )
                
        
        
        # iii. add on mass of entrants
        ike = np.sum(grid_k <= k_e) - 1     #grid index where k_e is located
        stationary_pdf[:,ike] = stationary_pdf[:,ike] + m*nu
            
        
        # iv. calculate supremum norm
        dist = np.abs(stationary_pdf-stationary_pdf_old).max()
        
        if dist < tol:
            break
        
        else:
            stationary_pdf_old = np.copy(stationary_pdf)
        
    return stationary_pdf, it, dist

#run everything

h_v3 = HopenhaynV3()
h_v3.solve_model()

