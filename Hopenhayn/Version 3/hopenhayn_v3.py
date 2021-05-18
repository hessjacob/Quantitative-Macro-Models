
"""
Author: Jacob Hess

Date: May 2021

Description: This code embeds the standard neoclassical growth model into the firm dynamics model from Hopenhayn (1992) and solves for the stationary 
equilibrium. Firms are heterogenous in productivity and own the capital stock thereby making the investment decision. Endogenous entry/exit  arises from 
the equilibrium solution and there are exogenous exit shocks which I include to induce larger/more productive firms to exit as well. Finally, there is a 
representative household who inelastically supplies labor. I calibrate the model to loosely replicate annual U.S. economy statistics. 

Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual)
    -- QuantEcon (to install: 'conda install quantecon')
"""

import time
import numpy as np
from numba import njit, prange
import quantecon as qe
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

    def __init__(self, cf = 1.5,              #fixed cost
                       ce = 3,                #entry cost
                       Nz = 20,                #number of discrete income states 
                       rho_z = 0.6,           #autocorrelation coefficient
                       sigma_z = 0.2,         #std. dev. of shocks
                       z_bar = 0,          #constant term in continuous income process (not the mean of the process)
                       plott =1,              #select 1 to make plots
                       ):
        
        #parameters subject to changes
        self.cf, self.ce, self.Nz, self.rho_z  = cf, ce, Nz, rho_z
        self.sigma_z, self.z_bar, self.plott = sigma_z, z_bar, plott 
        
        self.setup_parameters()
        self.setup_grid()
        self.setup_discretization()
        
        
    
    def setup_parameters(self):
        
        # a. model parameters
        self.beta = 0.9615        #annual discount factor 
        self.alpha = 0.85/3       #capital share
        self.gamma = 0.85*2/3     #labor share
        self.delta = 0.08         #annual depreciation rate
        self.lambdaa = 0.05       #exogenous exit rate
        self.psi = 0.5            #capital adjustment parameter
        self.xx = 1 - self.alpha - self.gamma   #for factor demand solution
        
        
        # b. incumbent firm soluton  
        self.tol = 1e-8                         #difference tolerance
        self.maxit = 2000                       #maximum value function iterations
        
        # capital grid
        self.Nk = 100       # number of capital grid points
        self.k_min = 0.01   # minimum capital level
        self.k_max = 250    # maximum capital level
        self.curv = 3       # grid curvature parameter
        
        # c. hh solution
        self.interest_rate = 1/self.beta - 1
        
    def setup_grid(self) :
        
        # a. capital grid
        self.grid_k = self.make_grid(self.k_min, self.k_max, self.Nk, self.curv)
        
    def setup_discretization(self):
        
        # a. discretely approximate the continuous tfp process 
        self.mc = qe.markov.approximation.rouwenhorst(self.Nz, self.z_bar, self.sigma_z, self.rho_z)
        #self.mc = qe.markov.approximation.tauchen(self.rho_z, self.sigma_z, self.z_bar, 3, self.Nz)

        # b. transition matrix and states
        self.pi = self.mc.P
        self.grid_z = np.exp(self.mc.state_values)
        
        # c. initial productivity distribution for entrant firm
        
        #For the initial distribution I use the stationary distribution of the transition matrix. 
        #I provide an alternative unfirom distribution. The results are very similar.
        
        self.nu = np.array(self.mc.stationary_distributions)
        #self.nu = (1/self.Nz)*np.ones(self.Nz)        #uniform distribution
        
        #the object mc returns nu as a 1d matrix. I convert this to a 1d array for the calcuations later.
        self.nu=np.squeeze(np.asarray(self.nu))
        
    
    
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
    
    
    def entrant_firm(self,VF):
    
        val = np.zeros(self.Nk)
        
        for ik in range(self.Nk):
            val[ik] = - self.grid_k[ik] + self.beta * np.dot(self.nu, VF[:,ik]) 
    
        VF_entrant = np.max(val)
        k_e = self.grid_k[np.argmax(val)]
        
        return VF_entrant, k_e
    
    
    
    def find_equilibrium_wage(self):
        """
        Using the bisection method this function finds the unique equilibrium wage that clears markets and satisfies the free entry condition. 
        
        In equilibrium the free entry condition (or the present discounted value) is zero. The free entry condition is where the expected firm value over 
        the initial productivity distribution equals the cost of entry (ce).
        """
        
        # a. set up the wage interval
        wmin, wmax = 0.01, 100
        
        for it_w in range(self.maxit):
                
            # i. guess a price
            wage_guess = (wmin+wmax)/2
            
            # ii. incumbent firm value function. present discounted value of incumbant
            VF = incumbent_firm(wage_guess, self.alpha, self.beta, self.delta, self.gamma, self.cf, self.psi, self.lambdaa, self.pi, 
                                self.grid_k, self.grid_z, self.Nz, self.Nk, self.maxit, self.tol)[0]

            # iii. entrant firm value function. present discounted value of a potential entrant
            
            VF_entrant = self.entrant_firm(VF)[0]
            free_entry_cond = VF_entrant - self.ce
            
            
            # iv. check if free entry condition is satisfied (present discounted value of a potential entrant equals zero)
            if np.abs(free_entry_cond) < self.tol:
                wage_ss = wage_guess
                break
            
            # v. update price interval
            if free_entry_cond < 0 :
                wmax=wage_guess 
            else:
                wmin=wage_guess
        
        return wage_ss

            
    
    
    
    
        
        
        
    #####################
    # 5. Main function #
    ####################
    
    def solve_model(self):
        """
        Finds the stationary equilibrium.
        """  
        
        t0 = time.time()    #start the clock
        
        
        # a. Find the steady state wage using bisection
        print('\nSolving firm problem...')
        
        self.wage_ss = self.find_equilibrium_wage()
        
        
        # b. Use the equilibrium wage to recover incumbent and entrant firm solutions
        self.VF, self.pol_k, self.pol_n, self.pol_enter, self.pol_inv, self.firm_output, self.it_vf = incumbent_firm(self.wage_ss, self.alpha,
                    self.beta, self.delta, self.gamma, self.cf, self.psi, self.lambdaa, self.pi, self.grid_k, self.grid_z, self.Nz,
                    self.Nk, self.maxit, self.tol)     
        
        self.VF_entrant, self.k_e = self.entrant_firm(self.VF)
        
        if self.it_vf < self.maxit:
            print(f"Value function convergence in {self.it_vf} iterations.")
        else : 
            print("Value function iteration: No convergence.")
        
        t1 = time.time()
        print(f'Firm problem time elapsed: {t1-t0:.2f} seconds')
        
        
        # c. Invariant joint distribution with endogenous exit
        print('\nFinding stationary distribution...')
        self.stat_dist_hat, self.it_d = solve_invariant_distribution(self.pol_k, self.grid_k, self.pol_enter, self.pi, self.nu, self.Nz,
                                                          self.Nk, self.maxit, self.lambdaa, self.tol)
        
        if self.it_d < self.maxit:
            print(f"Distribution convergence in {self.it_d} iterations.")
        else : 
            print("Distribution fixed point iteration: No convergence.")
        
        
        # d. Mass of entrants (m_star) in the ss equilibrium. Because labor is supplied inelastically we can use the labor market clearing condition
        # to solve m_star
        self.m_star = 1/np.tensordot(self.pol_n, self.stat_dist_hat)
        
        
        # e. Rescale to get invariant joint distribution (mass of plants)
        self.stat_dist = self.m_star * self.stat_dist_hat
        
        # invariant marginal productivity distribution by percent
        self.stat_dist_pdf = np.sum(self.stat_dist, axis=0) / np.sum(np.sum(self.stat_dist,axis=0))
        self.stat_dist_cdf = np.cumsum(self.stat_dist_pdf)
        
        t2 = time.time()
        print(f'Stationary distribution time elapsed: {t2-t1:.2f} seconds')
        
        
        # f. calculate employment distributions
        
        if self.plott:
            print("\nCalculating aggergate statistics and plotting...")
        else:
            print("\nCalculating aggergate statistics...")
        
        self.dist_emp = (self.pol_n * self.stat_dist_pdf)/ np.sum(self.pol_n * self.stat_dist_pdf)
        
        # invariant employment distribution by percent
        self.dist_emp_pdf = np.sum(self.pol_n * self.stat_dist_pdf, axis=0) / np.sum(np.sum(self.pol_n * self.stat_dist_pdf,axis=0))
        self.dist_emp_cdf = np.cumsum(self.dist_emp_pdf)
        
        
        # g. aggregate statistics
        
        self.Y_ss = np.tensordot(self.firm_output, self.stat_dist)
        self.Yfc_ss = np.tensordot(self.firm_output - self.cf, self.stat_dist)
        self.K_ss = np.tensordot(self.pol_k, self.stat_dist)
        self.N_ss = np.tensordot(self.pol_n, self.stat_dist) #should equal 1
        self.Inv_ss = np.tensordot(self.pol_inv, self.stat_dist)
        self.TFP_ss = self.Y_ss/(self.K_ss**self.alpha * self.N_ss**self.gamma)

        #use resource constraint to get aggregate consumption
        self.C_ss = self.Yfc_ss - self.Inv_ss + \
            np.tensordot(self.psi/2 * ((self.pol_k - (1-self.delta)*self.grid_k) / self.grid_k)**2 * self.grid_k, self.stat_dist) \
               - self.m_star*(self.ce+self.k_e)
        
        self.average_firm_size = self.N_ss / np.sum(np.sum(self.stat_dist,axis=0))
        self.exit_rate = 1 - np.sum((1-self.lambdaa)*np.dot(self.pi.T, self.stat_dist_hat)*self.pol_enter)/np.sum(self.stat_dist_hat) 
        
        
        # h. plot
        
        if self.plott:
            
            idx = [0, int(self.Nz/2), int(self.Nz-1)]
            for ii in idx :
                plt.plot(self.grid_k, self.VF[ii,:])
            plt.legend(['V(k,z$_{'+str(idx[0])+'}$)','V(k,z$_{'+str(idx[1])+'}$)', 'V(k,z$_{'+str(idx[2])+'}$)'])
            plt.title('Incumbant Firm Value Function')
            plt.xlabel('Capital (Firm Size)')
            #plt.savefig('vf_hopehaynv3.pdf')
            plt.show()
            
            for ii in idx :
                plt.plot(self.grid_k, self.pol_k[ii,:])
            plt.plot(self.grid_k, (1-self.delta)*self.grid_k,':')
            plt.legend(["k'(k,z$_{"+str(idx[0])+"}$)","k'(k,z$_{"+str(idx[1])+"}$)", "k'(k,z$_{"+str(idx[2])+"}$)", "k(1-$\delta$)"])
            plt.title('Capital Policy Funciton')
            plt.xlabel('Capital (Firm Size)')
            #plt.savefig('pol_k_hopehaynv3.pdf')
            plt.show()
            
            for ii in idx :
                plt.plot(self.grid_k, self.pol_inv[ii,:])
            plt.legend(['i(k,z$_{'+str(idx[0])+'}$)','i(k,z$_{'+str(idx[1])+'}$)', 'i(k,z$_{'+str(idx[2])+'}$)'])
            plt.title("Investment: $k'(k,z)-(1-\delta)k$")
            plt.xlabel('Capital (Firm Size)')
            #plt.savefig('pol_inv_hopehaynv3.pdf')
            plt.show()
            
            for ii in idx :
                plt.plot(self.grid_k, self.pol_n[ii,:])
            plt.legend(['n(k,z$_{'+str(idx[0])+'}$)','n(k,z$_{'+str(idx[1])+'}$)', 'n(k,z$_{'+str(idx[2])+'}$)'])
            plt.title("Labor Demand Policy Function")
            plt.xlabel('Capital (Firm Size)')
            #plt.savefig('pol_n_hopehaynv3.pdf')
            plt.show()
         
            plt.plot(self.grid_k, self.stat_dist_pdf)
            plt.plot(self.grid_k, self.dist_emp_pdf)
            plt.title('Stationary Marginal PDF' )
            plt.xlabel('Capital (Firm Size)')
            plt.ylabel('Density')
            plt.legend(['Productivity','Employment'])
            #plt.savefig('pdf_hopehaynv3.pdf')
            plt.show()
            
            plt.plot(self.grid_k, self.stat_dist_cdf)
            plt.plot(self.grid_k, self.dist_emp_cdf)
            plt.title('Stationary Marginal CDF' )
            plt.xlabel('Capital (Firm Size)')
            plt.ylabel('Cumulative Prob.')
            plt.legend(['Productivity','Employment'])
            #plt.savefig('cdf_hopehaynv3.pdf')
            plt.show()
            
        t3 = time.time()
        if self.plott : 
            print(f'Statistics and plot time elapsed: {t3-t2:.2f} seconds')
        else :
            print(f'Statistics time elapsed: {t3-t2:.2f} seconds')
        
        print("\n-----------------------------------------")
        print("Stationary Equilibrium")
        print("-----------------------------------------")
        print(f"ss wage  = {self.wage_ss:.2f}")
        print(f"exit rate = {self.exit_rate:.3f}")
        print(f"avg. firm size = {self.average_firm_size:.2f}")
        print(f"\nss output = {self.Y_ss:.2f}")
        print(f"ss investment = {self.Inv_ss:.2f}")
        print(f"ss tfp = {self.TFP_ss:.2f}")
        print(f"ss capital = {self.K_ss:.2f}")
        print(f"ss consumption = {self.C_ss:.2f}")
        print(f"ss labor = {self.N_ss:.2f}")
        
        t4 = time.time()
        print(f'\nTotal Run Time: {t4-t0:.2f} seconds')

        

#########################
# II. Jitted Functions #
########################

###########################################################
# 1. Incumbent firm problem and Value Function Iteration #
##########################################################

@njit(parallel=True)
def incumbent_firm(wage, alpha, beta, delta, gamma,  cf, psi, lambdaa, pi, grid_k, grid_z, Nz, Nk, maxit, tol):
    """
    Value function iteration for the incumbent firm problem.
    """ 

    
    #a. Initialize counters, initial guess. storage matriecs
    
    VF_old    = np.zeros((Nz,Nk))  #initial guess
    VF = np.copy(VF_old)
    pol_k = np.copy(VF_old)
    pol_n = np.copy(VF_old)
    pol_inv = np.copy(VF_old)
    pol_enter = np.copy(VF_old)
    firm_output = np.copy(VF_old)
    
    
    # c. given prices and hiring decision, iterate on incumbent firm vf
    for it in range(maxit):
        for iz in range(Nz):
            for ik in prange(Nk):
                
                pol_n[iz,ik] = ((grid_z[iz] * gamma) / wage)**(1/(1-gamma)) * grid_k[ik]**(alpha/(1-gamma))
                
                firm_output[iz,ik] = grid_z[iz] * grid_k[ik] ** alpha * pol_n[iz,ik] ** gamma
                
                firm_profit = firm_output[iz,ik]  - wage * pol_n[iz,ik] - (grid_k - (1-delta)*grid_k[ik]) \
                    - psi/2 * ((grid_k- (1-delta)*grid_k[ik]) / grid_k[ik])**2 * grid_k[ik] 
                    
                val = firm_profit + beta * (1-lambdaa)*np.dot(pi[iz,:], VF_old)
                
                VF[iz,ik] = np.maximum( (1-delta) * grid_k[ik] - (psi * (1-delta)**2 * grid_k[ik])/2, np.max(val) - cf)
                
                if VF[iz,ik] == (1-delta) * grid_k[ik] - (psi * (1-delta)**2 * grid_k[ik])/2 : 
                    pol_k[iz,ik] = 0
                    pol_inv[iz,ik] = 0
                    
                else: 
                    pol_k[iz,ik] = grid_k[np.argmax(val)]
                    pol_enter[iz,ik] = 1
                    pol_inv[iz,ik] = pol_k[iz,ik] - (1-delta) * grid_k[ik]
                
        
        
        dist = np.abs(VF - VF_old).max()
    
        if dist < tol :
           break
       
        VF_old = np.copy(VF)


    return VF, pol_k, pol_n, pol_enter, pol_inv, firm_output, it



####################################
# 2. Find stationary distribution #
###################################

@njit(parallel=True)
def solve_invariant_distribution(pol_k, grid_k, pol_enter, pi, nu, Nz, Nk, maxit, lambdaa, tol):
    
    """
    Fixed point iteration method to approximates the stationary joint distribution of firms. The function iterates on the law
    of motion. The mass of entrants (m) is normalized to one.
    """
    
    # a. initialize
    stat_dist_0 = np.random.uniform(0,1, size=(Nz, Nk)) #initial guess
    stat_dist_0 = stat_dist_0 / np.sum(stat_dist_0)
    stat_dist_hat = np.zeros((Nz,Nk))
    m = 1   #we normalize the mass of potential entrants to one
    
    # b. fixed point iteration
    
    for it_d in range(maxit):
        
        for ik in range(Nz) :
            for il in prange(Nk):
                
                indict_k = (pol_k == grid_k[il])
                
                summ = 0
                
                for ii in prange(Nz) :
                    for ij in prange(Nk) :
                        
                        sum_ij = (1-lambdaa) * pi[ii,ik] * indict_k[ii,ij] * pol_enter[ii, ij] * stat_dist_0[ii, ij] + m * pol_enter[ii, ij] * nu[ik]
                        summ += sum_ij
                
                stat_dist_hat[ik, il] = summ
                
        
        dist = np.abs(stat_dist_hat - stat_dist_0).max()
        
        if dist < tol:
            break
        
        else: 
           stat_dist_0 = stat_dist_hat
    
            
    return stat_dist_hat, it_d


#run everything

h_v3 = HopenhaynV3()
h_v3.solve_model()








