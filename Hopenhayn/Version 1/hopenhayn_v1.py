
"""
Author: Jacob Hess

Date: January 2021

Description: Replicates Hopenhayn (1992) which finds the stationary equilibrium in a firm dynamics model. I follow the 
algorithm laid out by Gianluca Violante's and Chris Edmond's notes where I solve the incumbant firm problem with value 
function itertation and analytically solve for the stationary distribution. I use Chris Edmond's calibration from his 
example in ch. 3. A slight difference in my code is that I use the rouwenhorst method instead of the tauchen to make 
the continuous tfp process discrete but I still use 20 nodes as in the paper.   

Aknowledgements: I used notes or pieces of code from the following :
    1) Gianluca Violante's notes (https://sites.google.com/a/nyu.edu/glviolante/teaching/quantmacro)
    2) Chris Edmond's notes (http://www.chrisedmond.net/phd2014.html)
    
Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual)
    -- QuantEcon (to install: 'conda install quantecon')
"""


import time
import numpy as np
import numpy.linalg as la
import quantecon as qe
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')


class Hopenhayn:
    
    """
    Class object of the model. Hopenhayn().solve_model() runs everything
    """

    ############
    # 1. setup #
    ############

    def __init__(self, beta=0.8,             #discount factor 5 year
                        theta=2/3,               #labor share
                       cf=20,                    #fixed cost 
                       ce=40,                   #entry cost
                       D = 100,                  #size of the market (exogeneous)
                       wage = 1,                   #wage, normalized to one
                       rho_z = 0.9,           #autocorrelation coefficient
                       sigma_z = 0.2,         #std. dev. of shocks
                       Nz = 20,                #number of discrete income states
                       z_bar = 0.14,          #constant term in continuous income process (not the mean of the process)
                       plott =1,              #select 1 to make plots
                       ):
        
        #parameters subject to changes
        self.beta, self.theta, self.cf, self.ce, self.D, self.wage, self.rho_z  = beta, theta, cf, ce, D, wage, rho_z
        self.sigma_z, self.Nz, self.z_bar, self.plott = sigma_z, Nz, z_bar, plott 
        
        self.setup_parameters()
        self.setup_grid()
        
        
    
    def setup_parameters(self):
        
        # a. incumbent firm soluton  
        self.tol = 1e-8          #difference tolerance
        self.maxit = 2000        #maximum value function iterations
        
    def setup_grid(self):
        
        # a. discretely approximate the continuous tfp process 
        self.mc = qe.markov.approximation.rouwenhorst(self.Nz, self.z_bar, self.sigma_z, self.rho_z)
        #self.mc = qe.markov.approximation.tauchen(self.rho_z, self.sigma_z, self.z_bar, 3, self.Nz)

        # b. transition matrix and states
        self.pi = self.mc.P
        self.grid_z = np.exp(self.mc.state_values)
        
        # c. initial productivity distribution for entrant firm
        
        # there is no explicit assumption on the distribution in hopenhayn or edmond's notes.
        # I use the stationary distribution of the transition matrix. I provide an alternative
        # unfirom distribution. The results are very similar.
        
        self.nu = np.array(self.mc.stationary_distributions)
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
    
    def static_profit_max(self, price):
        """
        Static incumbent firm profit maximzation.
        """
        
        # a. given prices, find the labor hiring decision policy function (analytical expression derived in 
        #violante's notes pg. 2)
        pol_n = ((self.theta * price * self.grid_z) / self.wage) ** (1 / (1 - self.theta))
        
        # b. given prices and hiring decision, find firm output
        firm_output = self.grid_z * (pol_n ** self.theta)
        
        # c. given prices and hiring decision, find profits by solving static firm problem
        firm_profit = price*firm_output - self.wage*pol_n - self.cf
        
        return firm_profit, firm_output, pol_n
    
    def incumbent_firm(self, price):
        """
        Value function iteration for the incumbent firm problem.
        """ 

        # a. initialize 
        VF_old = np.zeros(self.Nz)
        VF = np.zeros(self.Nz)
        
        # b. solve the static firm problem
        firm_profit, firm_output, pol_n = self.static_profit_max(price)
    
        # c. given prices and hiring decision, iterate on incumbent firm vf
        for it in range(self.maxit):
            
            VF = firm_profit + self.beta * np.dot(self.pi, VF_old).clip(min=0)
            
            dist = np.abs(VF_old - VF).max()
        
            if dist < self.tol :
               break
           
            VF_old = np.copy(VF)

        # d. enter/stay in the market policy function 
        pol_enter = np.ones(self.Nz)*(VF>0)
        
        # e. productivity exit threshold
        idx = np.searchsorted(pol_enter, 1) #index of self.pol_enter closest to one on the left
        exit_cutoff = self.grid_z[idx]
        
        # f. alternative way to do steps d and e
        #avg_VF = np.dot(self.pi, VF)
        #idx = np.searchsorted(avg_VF, 0) #index of avg_VF closest to zero on the left
        
        #exit_cutoff = self.grid_z[idx]
        #pol_exit = np.where(self.grid_z < exit_cutoff, 1, 0)
        #pol_enter = 1 - pol_exit

        return VF, firm_profit, firm_output, pol_n, pol_enter, exit_cutoff
    
    
    
    ####################################
    # 4. Find stationary equilibrium  #
    ###################################
    
    
    def find_equilibrium_price(self):
        """
        Finds the equilibrium price that clears markets. 
        
        The function follows steps 1-3 in algorithm using the bisection method. It guesses a price, solves incumbent firm vf 
        then checks whether free entry condition is satisfied. If not, updates the price and tries again. The free entry condition 
        is where the firm value of the entrant (VF_entrant) equals the cost of entry (ce) and hence the difference between the two is zero. 
        """
        
        # a. initial price interval
        pmin, pmax = 1, 100
        
        # b. iterate to find prices
        for it_p in range(self.maxit):
            
            # i. guess a price
            price = (pmin+pmax)/2
            
            # ii. incumbent firm value function
            VF = self.incumbent_firm(price)[0]
        
            # iii. entrant firm value function
            VF_entrant = self.beta * np.dot(VF, self.nu)
            
            # iv. check if free entry condition is satisfied
            diff = np.abs(VF_entrant-self.ce)
            
            if diff < self.tol:
                break
            
            # v. update price interval
            if VF_entrant < self.ce :
                pmin=price 
            else:
                pmax=price
        
        return price
            
    
    def solve_invariant_distribution(self, m, pol_enter):
        pi_tilde = (self.pi * pol_enter.reshape(self.Nz, 1)).T
        I = np.eye(self.Nz)
         
        return m * ( np.dot( la.inv(I - pi_tilde), self.nu ) )
    
    
        
        
    #####################
    # 5. Main function #
    ####################
    
    def solve_model(self):
        """
        Finds the stationary equilibrium
        """  
        
        t0 = time.time()    #start the clock
        
        # a. Find the optimal price using bisection (algo steps 1-3)
        self.price_ss = self.find_equilibrium_price()
        
        # b. Use the equilibrium price to recover incumbent firm solution
        self.VF, self.firm_profit, self.firm_output, self.pol_n, self.pol_enter, self.exit_cutoff = self.incumbent_firm(self.price_ss)
        
        # c. Invariant (productivity) distribution with endogenous exit. Here assume m=1 which 
        #will come in handy in the next step.
        self.distrib_stationary_0 = self.solve_invariant_distribution(1, self.pol_enter)
        
        # d. Rather than iterating on market clearing condition to find the equilibrium mass of entrants (m_star)
        # we can compute it analytically (Edmond's notes ch. 3 pg. 25)
        self.m_star = self.D / ( np.dot( self.distrib_stationary_0, self.firm_output) )
        
        # e. Rescale to get invariant (productivity) distribution (mass of plants)
        self.distrib_stationary = self.m_star * self.distrib_stationary_0
        self.total_mass = np.sum(self.distrib_stationary)
        
        # Invariant (productivity) distribution by percent
        self.pdf_stationary = self.distrib_stationary / self.total_mass
        self.cdf_stationary = np.cumsum(self.pdf_stationary)
        
        # f. calculate employment distributions
        self.distrib_emp = (self.pol_n * self.distrib_stationary)
        
        # invariant employment distribution by percent
        self.pdf_emp = self.distrib_emp / np.sum(self.distrib_emp)
        self.cdf_emp = np.cumsum(self.pdf_emp)
        
        # g. calculate statistics
        self.total_employment = np.dot(self.pol_n, self.distrib_stationary)
        self.average_firm_size = self.total_employment / self.total_mass
        self.exit_rate = self.m_star / self.total_mass
        #self.exit_rate = 1-(np.sum(self.pi.T*self.distrib_stationary_0*self.pol_enter)/np.sum(self.distrib_stationary_0)) #alternative calculation
        
        # h. plot
        
        if self.plott:
            plt.plot(self.grid_z, self.VF)
            plt.axvline(self.exit_cutoff, color='tab:red', linestyle='--', alpha=0.7)
            plt.axhline(0, color='tab:green', linestyle='--', alpha=0.7)
            plt.title('Incumbant Firm Value Function')
            plt.legend(['Value Function', 'Exit Threshold='+str(self.exit_cutoff.round(2)),'VF <= 0'])
            plt.xlabel('Productivity level')
            #plt.savefig('value_func_hopehayn.pdf')
            plt.show()
         
            plt.plot(self.grid_z,self.pdf_stationary)
            plt.plot(self.grid_z, self.pdf_emp)
            plt.title('Stationary PDF' )
            plt.xlabel('Productivity level')
            plt.ylabel('Density')
            plt.legend(['Share of Firms','Share of Employment'])
            #plt.savefig('pdf_hopehayn.pdf')
            plt.show()
            
            plt.plot(self.grid_z,self.cdf_stationary)
            plt.plot(self.grid_z, self.cdf_emp)
            plt.title('Stationary CDF' )
            plt.xlabel('Productivity level')
            plt.ylabel('Cumulative Sum')
            plt.legend(['Share of Firms','Share of Employment'])
            #plt.savefig('cdf_hopehayn.pdf')
            plt.show()
            
            #employment share pie charts 
            employed = [20, 50, 100, 500]
            
            #percentage of firms that employ employed
            self.share_firms = np.zeros(len(employed)+1)
            for i in range(len(employed)):
                summ = np.sum(self.share_firms)
                interpolate = self.interpol(self.pol_n, self.cdf_stationary, employed[i])[0]
                self.share_firms[i] = interpolate - summ
            self.share_firms[-1] = 1 - np.sum(self.share_firms)
            
            plt.pie(self.share_firms, labels=['<20','21<50','51<100','101<500','501<'], autopct="%.1f%%")
            plt.title('Size of Firms by Number of Employees')
            #plt.savefig('firm_size_hopehayn.pdf')
            plt.show()
            
            self.share_employment = np.zeros(len(employed)+1)
            
            for i in range(len(employed)):
                summ = np.sum(self.share_employment)
                interpolate = self.interpol(self.pol_n, self.cdf_emp, employed[i])[0]
                self.share_employment[i] = interpolate - summ
            self.share_employment[-1] = 1 - np.sum(self.share_employment)
            
            plt.pie(self.share_employment, labels=['<20','21<50','51<100','101<500','501<'], autopct="%.1f%%")
            plt.title('Employment Share by Firm Size')
            #plt.savefig('employment_by_firm_size_hopehayn.pdf')
            plt.show()
            
            #these pie sharts show that most firms are small, few large firms. In the second it says most people 
            #are employed by large firms
        
        print("\n-----------------------------------------")
        print("Stationary Equilibrium")
        print("-----------------------------------------")
        print(f"ss price  = {self.price_ss:.2f}")
        print(f"entry/exit rate = {self.exit_rate:.3f}")
        print(f"avg. firm size = {self.average_firm_size:.2f}")
        
        t1 = time.time()
        print(f'\nTotal Run Time: {t1-t0:.2f} seconds')

        
#run everything

hvfi=Hopenhayn()
hvfi.solve_model()








