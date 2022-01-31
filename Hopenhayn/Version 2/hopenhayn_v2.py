
"""
Author: Jacob Hess

Date: May 2021

Description: This code embeds the standard neoclassical growth model into the firm dynamics model from Hopenhayn (1992) and solves for the stationary equilibrium. 
There is a representative household who inelastically supplies labor and rents capital to firms. Firms are heterogenous in productivity and are subject to 
idiosyncratic shocks each period and yhere is endogenous entry/exit which arises from the equilibrium solution. In addition, there are exogenous exit shocks which 
I include to induce larger/more productive firms to exit as well. I calibrate the model to loosely replicate annual U.S. economy statistics. 

Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual)
    -- QuantEcon (to install: 'conda install quantecon')
"""

import time
import numpy as np
from scipy import stats
import quantecon as qe
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

class HopenhaynV2:
    
    """
    Class object of the model. HopenhaynV2().solve_model() runs everything
    """

    ############
    # 1. setup #
    ############

    def __init__(self, cf = 1,              #fixed cost
                       ce = 10,             #entry cost
                       rho_z = 0.6,           #autocorrelation coefficient
                       sigma_z = 0.2,         #std. dev. of shocks
                       Nz = 20,                #number of discrete income states
                       z_bar = 0,          #constant term in continuous income process (not the mean of the process)
                       plott =1,              #select 1 to make plots
                       ):
        
        #parameters subject to changes
        self.cf, self.ce, self.Nz, self.rho_z  = cf, ce, Nz, rho_z
        self.sigma_z, self.z_bar, self.plott = sigma_z, z_bar, plott 
        
        self.setup_parameters()
        self.setup_grid()
        
        
    
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
        
        # c. hh solution
        self.interest_rate = 1/self.beta - 1
        self.rental_rate = self.interest_rate + self.delta
        
    def setup_grid(self):
        
        # a. discretely approximate the continuous tfp process 
        self.mc = qe.markov.approximation.rouwenhorst(self.Nz, self.z_bar, self.sigma_z, self.rho_z)
        #self.mc = qe.markov.approximation.tauchen(self.rho_z, self.sigma_z, self.z_bar, 3, self.Nz)

        # b. transition matrix and states
        self.pi = self.mc.P
        self.grid_z = np.exp(self.mc.state_values)
        
        # c. initial productivity distribution for entrant firm
        
        # Following Ranasinghe (2014,2017) I assime that nu is normally distributed and set the mean to 1.3 and variance to 0.22 to match the size distribution of 
        #entering firms (as per the Statistics of US Business, 2010). The calibrated values imply that 85 (95) percent of entering firms have fewer than 10 (20) 
        #employees. I include two additional alternatives for the initial distribution. One is the stationary distribution of the transition matrix. Another is the
        # unfirom distribution. The results are very similar for all.
        
        self.mu_enter = 1.3
        self.sigma_enter = 0.22
        
        self.nu_cdf = stats.norm.cdf(self.grid_z, self.mu_enter, self.sigma_enter)
        self.nu = np.insert(np.diff(self.nu_cdf), 0 , self.nu_cdf[0])
        
        #self.nu = np.array(self.mc.stationary_distributions)
        #self.nu = (1/self.Nz)*np.ones(self.Nz)        #uniform distribution
        
        #the object mc returns nu as a 1d matrix. I convert this to a 1d array for the calcuations later.
        self.nu=np.squeeze(np.asarray(self.nu))
        
    
    
    ####################################
    # 2. Helper functions #
    ####################################
    
    
    
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
    


    ################################################
    # 3. Solve incumbent and entrant firm problem #
    ###############################################
    
    
    def static_profit_max(self, wage):
        """
        Static incumbent firm profit maximzation given prices (wage and interest rate). Given prices, the function returns firm revenues, 
        profits and factor demands.
        """

        # a. factor demands
        
        # i. optimal capital demand
        pol_k = self.grid_z**(1/self.xx) * (self.alpha /self.rental_rate)**((1-self.gamma)/self.xx) * (self.gamma / wage)**(self.gamma/self.xx)
        
        # ii. optimal labor demand
        pol_n = self.grid_z**(1/self.xx) * (self.alpha /self.rental_rate)**(self.alpha/self.xx) * (self.gamma / wage)**((1-self.alpha)/self.xx)
        
        # b. firm revenues
        firm_output = self.grid_z * pol_k ** self.alpha * pol_n ** self.gamma
        
        # c. firm profits
        firm_profit = firm_output - wage*pol_n - self.rental_rate*pol_k - self.cf
        
        return firm_profit, firm_output, pol_k, pol_n
    
    
    
    def incumbent_firm(self, wage):
        """
        Value function iteration for the incumbent firm problem.
        """ 

        # a. initialize 
        VF_old = np.zeros(self.Nz)
        VF = np.zeros(self.Nz)
        
        # b. solve the static firm problem
        firm_profit = self.static_profit_max(wage)[0]
    
        # c. given prices and hiring decision, iterate on incumbent firm vf
        for it in range(self.maxit):
            
            VF = firm_profit + (1-self.lambdaa) * self.beta * np.dot(self.pi, VF_old)
            
            VF = VF*(VF>0)
            
            dist = np.abs(VF - VF_old).max()
        
            if dist < self.tol :
               break
            
            else:
                VF_old = np.copy(VF)

        # d. enter/stay in the market policy function 
        pol_enter = np.ones(self.Nz)*(VF>0)

        return VF, pol_enter
    
    
    
    def find_equilibrium_wage(self):
        """
        Using the bisection method this function finds the unique equilibrium wage that clears markets and satisfies the free entry condition. 
        
        In equilibrium the free entry condition (or the present discounted value of the potential entrant) is zero. The free entry condition is where the expected firm value over 
        the initial productivity distribution equals the cost of entry (ce).
        """
        
        # a. set up the wage interval
        wmin, wmax = 0.01, 100
        
        # i. ensure that wmin is low enough for bisection to work
        VF_min = self.incumbent_firm(wmin)[0]
        VF_entrant_min = self.beta*np.dot(VF_min, self.nu) - self.ce  #present discounted value of a potential entrant
        
        assert VF_entrant_min > 0, 'wmin is set too high.'
        
        
        # ii. ensure that wmax is high enough for bisection to work. In case that it is not this runs a loop to find a high enough bound
            
        for i_pv in range(self.maxit):
            VF_max = self.incumbent_firm(wmax)[0]
            PV_entrant_max = self.beta*np.dot(VF_max, self.nu) - self.ce  #present discounted value of a potential entrant
            
            if PV_entrant_max < 0:
                break
            
            else:
                wmax += 100
                
        assert PV_entrant_max < 0, 'wmax, can not find upper bound of wage high enough for bisection to work => No convergence'
        
        
        
        # b. iterate to find wage
        for it_w in range(self.maxit):
            
            # i. guess a price
            wage_guess = (wmin+wmax)/2
            
            # ii. incumbent firm value function. present discounted value of incumbant
            VF = self.incumbent_firm(wage_guess)[0]
        
            # iii. Free entry condition or present discounted value of a potential entrant. 
            free_entry_cond = self.beta*np.dot(VF, self.nu) - self.ce
            
            # iv. check if free entry condition is satisfied (present discounted value of a potential entrant equals zero)
            if np.abs(free_entry_cond) < self.tol:
                wage_ss = wage_guess
                break
            
            # v. update price interval
            if free_entry_cond < 0 :
                wmax=wage_guess 
            else:
                wmin=wage_guess
        
            #altnatively could do
            #if free_entry_cond * PV_entrant_max  > 0:
            #    wmax = wage_guess
            #else :
            #    wmin = wage_guess
        
        return wage_ss
            
    
    
    
    ###################################################
    # 4. Find stationary (productivity) distribution #
    ##################################################
    
    def solve_invariant_distribution(self):
        """
        Solves for the stationary (productivty) distribution by fixed point iteration. The mass of entrants (m) is normalized to one.
        """
        
        # a. initialize
        stat_dist_0 = np.zeros(self.Nz) #initial guess
        m = 1   #we normalize the mass of potential entrants to one
        
        # b. fixed point iteration
        
        for it_d in range(self.maxit):
            stat_dist_hat = (1-self.lambdaa)*np.dot(stat_dist_0, self.pi*self.pol_enter) + m * self.nu * self.pol_enter
            dist = np.abs(stat_dist_hat - stat_dist_0).max()
            
            if dist < self.tol:
                break
            
            else: 
                stat_dist_0 = stat_dist_hat
        
                
        return stat_dist_hat
        
        
        
    #####################
    # 5. Main function #
    ####################
    
    def solve_model(self):
        """
        Finds the stationary equilibrium.
        """  
        
        t0 = time.time()    #start the clock
        
        # a. Find the steady state wage using bisection
        self.wage_ss = self.find_equilibrium_wage()
        
        
        # b. Use the equilibrium wage to recover incumbent firm solution
        self.firm_profit, self.firm_output, self.pol_k, self.pol_n = self.static_profit_max(self.wage_ss)
        self.VF, self.pol_enter = self.incumbent_firm(self.wage_ss)     
        
        
        # c. Invariant (productivity) distribution with endogenous exit
        self.stat_dist_hat = self.solve_invariant_distribution()
        
        
        # d. Mass of entrants (m_star) in the ss equilibrium. Because labor is supplied inelastically we can use the labor market clearing condition
        # to solve m_star, This uses the condition that aggregate labor demand is equal to 1 = N_ss = m_star*np.dot(self.stat_dist_hat, self.pol_n)
        self.m_star = 1/np.dot(self.stat_dist_hat, self.pol_n)
        
        # e. Rescale to get invariant (productivity) distribution (mass of plants)
        self.stat_dist = self.m_star * self.stat_dist_hat
        
        # invariant (productivity) distribution by percent
        self.stat_dist_pdf = self.stat_dist / np.sum(self.stat_dist)
        self.stat_dist_cdf = np.cumsum(self.stat_dist_pdf)
        
        # f. calculate employment distributions
        self.dist_emp = (self.pol_n * self.stat_dist)
        
        # invariant employment distribution by percent
        self.dist_emp_pdf = self.dist_emp / np.sum(self.dist_emp)
        self.dist_emp_cdf = np.cumsum(self.dist_emp_pdf)
        
        # g. aggregate statistics
        
        self.Y_ss = np.dot(self.firm_output, self.stat_dist)
        #self.Y_ss = np.sum(self.firm_output*self.stat_dist, axis=0) alternative way to calculate it
        self.Yfc_ss = np.dot(self.firm_output - self.cf, self.stat_dist)
        self.K_ss = np.dot(self.pol_k, self.stat_dist)
        self.N_ss = np.dot(self.pol_n, self.stat_dist)  #should equal 1
        self.profit_ss = np.dot(self.firm_profit, self.stat_dist)
        self.TFP_ss = self.Y_ss/(self.K_ss**self.alpha * self.N_ss**self.gamma)
        
        #use resource constraint to get aggregate consumption
        self.C_ss = self.Yfc_ss - self.delta*self.K_ss - self.ce*self.m_star
        
        self.average_firm_size = self.N_ss / np.sum(self.stat_dist)
        #np.dot(self.stat_dist_pdf, self.pol_n)  #alternative calculation
        self.exit_rate = 1 - np.sum((1-self.lambdaa)*np.dot(self.pi.T, self.stat_dist_hat)*self.pol_enter)/np.sum(self.stat_dist_hat) 
        #self.exit_rate_alt = self.m_star / np.sum(self.stat_dist)    #alternative calculation
        
        # h. plot
        
        if self.plott:
            
            idx = np.searchsorted(self.pol_enter, 0, side='right') #producitivity threshold at which to exit
            self.exit_cutoff = self.grid_z[idx]  #exit threshold productivity value
            
            plt.plot(self.grid_z, self.VF)
            plt.axvline(self.exit_cutoff, color='tab:red', linestyle='--', alpha=0.7)
            plt.title('Incumbant Firm Value Function')
            plt.legend(['Value Function', 'Exit Threshold='+str(self.exit_cutoff.round(2))])
            plt.xlabel('Productivity level')
            #plt.savefig('vf_hopehaynv2.pdf')
            plt.show()
         
            plt.plot(self.grid_z,self.stat_dist_pdf)
            plt.plot(self.grid_z, self.dist_emp_pdf)
            plt.title('Stationary PDF' )
            plt.xlabel('Productivity level')
            plt.ylabel('Percent')
            plt.legend(['Productivity','Employment'])
            #plt.savefig('pdf_hopehaynv2.pdf')
            plt.show()
            
            plt.plot(self.grid_z, self.stat_dist_cdf)
            plt.plot(self.grid_z, self.dist_emp_cdf)
            plt.title('Stationary CDF' )
            plt.xlabel('Productivity level')
            plt.ylabel('Cumulative Sum')
            plt.legend(['Productivity','Employment'])
            #plt.savefig('cdf_hopehaynv2.pdf')
            plt.show()
            
            #employment share pie charts 
            employed = [5, 10, 20, 100]
            
            #percentage of firms that employ employed
            self.share_firms = np.zeros(len(employed)+1)
            for i in range(len(employed)):
                summ = np.sum(self.share_firms)
                interpolate = self.interpol(self.pol_n, self.stat_dist_cdf, employed[i])[0]
                self.share_firms[i] = interpolate - summ
            self.share_firms[-1] = 1 - np.sum(self.share_firms)
            
            plt.pie(self.share_firms, labels=['$\leq$5','6$\leq$10','11$\leq$20','21$\leq$100','101$\leq$'], autopct="%.1f%%")
            plt.title('Firm Size')
            #plt.savefig('firm_size_hopehaynv2.pdf')
            plt.show()
            
            self.share_employment = np.zeros(len(employed)+1)
            
            for i in range(len(employed)):
                summ = np.sum(self.share_employment)
                interpolate = self.interpol(self.pol_n, self.dist_emp_cdf, employed[i])[0]
                self.share_employment[i] = interpolate - summ
            self.share_employment[-1] = 1 - np.sum(self.share_employment)
            
            plt.pie(self.share_employment, labels=['$\leq$ 5','6$\leq$10','11$\leq$20','21$\leq$100','101$\leq$'], autopct="%.1f%%")
            plt.title('Share of Employment')
            #plt.savefig('employment_share_hopehaynv2.pdf')
            plt.show()
            
            #these pie sharts show that most firms are small, few large firms. In the second it says most people 
            #are employed by medium and large firms
        
        print("\n-----------------------------------------")
        print("Stationary Equilibrium")
        print("-----------------------------------------")
        print(f"ss wage  = {self.wage_ss:.2f}")
        print(f"entry/exit rate = {self.exit_rate:.3f}")
        print(f"avg. firm size = {self.average_firm_size:.2f}")
        print(f"\nss output = {self.Y_ss:.2f}")
        print(f"ss tfp = {self.TFP_ss:.2f}")
        print(f"ss capital = {self.K_ss:.2f}")
        print(f"ss consumption = {self.C_ss:.2f}")
        print(f"ss labor = {self.N_ss:.2f}")
        print(f"ss profit = {self.profit_ss:.2f}")
        
        t1 = time.time()
        print(f'\nTotal Run Time: {t1-t0:.2f} seconds')

        
#run everything

h_v2_ngm=HopenhaynV2()
h_v2_ngm.solve_model()








