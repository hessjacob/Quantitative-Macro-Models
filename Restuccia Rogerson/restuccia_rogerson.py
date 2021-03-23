"""
Author: Jacob Hess

Date: March 2021

Description: Replicates Restuccia and Rogerson (2008) which shows that resource misallocation across heterogenous firms 
can have sizeable effects on aggregate output and TFP. The code calculates the efficient or benchmark economy and then compares 
economies under a policy distortion of either output, capital or labor which reallocates resources among firms through tax/subsidies. Each firm faces
its own tax or subdidy. The focus is on policies that create idiosyncratic distortions to establishment-level decisions and hence cause a reallocation
of resources across establishments. To emphasize the impact of misallocation alone on output and TFP, policy does not relay on aggregate capital accumulation or 
aggregate price differences, firm productivity is exogenous and constant and entry nor the distributions are affected by distortions. 
With that said, for each tax rate the code finds the subsidy rate that will generate the same aggregate capital stock as the benchmark economy. 

By default the code replicates Tables 3 and 5 in the paper. 
    
Aknowledgements: The code is loosely based on the paper's code:
    1) Restuccia and Rogerson (2008) code -- https://ideas.repec.org/c/red/ccodes/07-48.html
    
Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual)
    -- Tabulate (to install: 'pip install tabulate')
    
Required data files: 
    -- establishment_dist.txt, firm size by number of employees which is used to construct productivity cdf.
"""

import time
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


class RestucciaRogerson:
    
    """
    Class object of the model. Hopenhayn().solve_model() runs everything
    """

    ############
    # 1. setup #
    ############

    def __init__(self, subsidy_frac = 0.5,   #fraction of population that recieves subsidy. to replicate tables 4 and 6 adjust this parameter
                       excempt_frac =0,    #fraction of population excempt from taxation
                       distortion_case = 1,      #tax/subsidy redistribution. see description below
                       policy_type = 1,      #tax/subsidy type. 1 for output, 2 for capital, 3 for labor 
                       tau_vector = [0.1, 0.2, 0.3, 0.4],    #tax levels to test
                       beta=0.96,           #discount factor 1 year
                       alpha = 0.85/3,       #capital share
                       gamma = 0.85*2/3,     #labor share
                       cf=0,                #normalized fixed cost 
                       ce=1,                #entry cost (set to zero for benchmark case)
                       delta = 0.08,         #annual depreciation rate
                       lambdaa = 0.1,        #annual exit rate
                       Ns = 100,            #number of discrete productivity states
                       plott =1,              #select 1 to make plots
                       ):
        
        #parameters subject to changes
        self.beta, self.alpha, self.gamma, self.cf, self.ce, self.delta, self.lambdaa, self.Ns, self.plott = beta, alpha, gamma, cf, ce, delta, lambdaa, Ns, plott
        self.subsidy_frac, self.excempt_frac, self.distortion_case, self.policy_type, self.tau_vector = subsidy_frac, excempt_frac, distortion_case, policy_type, tau_vector
        
        self.setup_parameters()
        self.setup_grid()
        
        #warnings
        assert self.distortion_case == 1 or self.distortion_case == 2 or self.distortion_case == 3, 'Choose 1, 2, or 3 for distortion_case'
        assert self.policy_type == 1 or self.policy_type == 2 or self.policy_type == 3, 'Choose 1, 2, or 3 for spolicy_type'
        assert self.subsidy_frac + self.excempt_frac + (1 - self.subsidy_frac - self.excempt_frac) == 1, 'Percentage of those subsidized, excempt and taxed exceeds 1'
    
    
    
    
    
    def setup_parameters(self):
        
        #a. constant prices
        self.ret = 1 / self.beta - 1 + self.delta    #return on capital  
        
        self.interest_rate = self.ret - self.delta       #real interest rate (recall in complete markets (1+i)beta = 1)
        
        #b. firm's discount rate
        self.rho = (1 - self.lambdaa )/(1 + self.interest_rate)
        
        #c. steady state solution
        self.ntau = 3   #number of categories for tau: I assume subsidy, excempt, taxed
        self.big_S = 1  #scaling parameter
        
        #d. fixed point iteration  
        self.tol = 1e-8          #difference tolerance
        self.maxit = 100        #maximum bisection iterations
        
        #bisection guesses
        self.w0_guess = 0.5
        self.w1_guess = 20
        self.tau_s_0 = 0.0
        self.tau_s_1 = 1.2
        
        #e. distribution storage
        self.prod_pdf = np.zeros(self.Ns) # pdf over the idiosyncratic draws of s. denoted h in paper
        
        #pdf (over tau and s) that a establishment with productivity s faces policy tau. the possibility that the value of the firm tax rate may be correlated with the draw of the firm productivity from h(s)
        self.policy_pdf = np.vstack((self.subsidy_frac*np.ones(self.Ns), self.excempt_frac*np.ones(self.Ns), (1-self.excempt_frac-self.subsidy_frac)*np.ones(self.Ns))).T     

        #f. policy evaluation storage
        self.Yss_d = np.zeros(len(self.tau_vector))
        self.TFPss_d = np.zeros(len(self.tau_vector))
        self.Kss_d =  np.zeros(len(self.tau_vector))
        self.E_star_d =  np.zeros(len(self.tau_vector))
        self.Y_set_d = np.zeros((len(self.tau_vector),self.ntau))
        self.subsidy_size_d =  np.zeros(len(self.tau_vector))
        self.w_ss_d =  np.zeros(len(self.tau_vector))
        self.N_ss_d =  np.zeros(len(self.tau_vector))
        self.average_firm_size_d = np.zeros(len(self.tau_vector))
        self.tau_s = np.zeros(len(self.tau_vector))
        
        self.cdf_stationary_d = np.zeros((4,100))
        self.cdf_emp_d = np.zeros((4,100))
    
        
        
    
    
    def setup_grid(self):
        
        #a. productivity state grid
        
        #In the benchmark economy there is a simple mapping between establishment-level productivity and employment.
        
        #i. pdf of number of employees, ranging from 1 to 10,000
        df = pd.read_csv('establishment_dist.txt', sep='\t',header=None, names=['upper_s', 'hs'])    #upper_s, (upper range of number of employees) hs, fraction of firm in each group
        df['Hs'] = np.cumsum(df.hs)

        #ii. get maximum s value usingthe relationship with relative labor demand (s_min is normalized to 1)
        s_max = df.loc[df.index[-1], 'upper_s'] ** (1-self.gamma-self.alpha) 
        
        #iii. make the grid. ranges from 1 to 3.98
        self.grid_s = np.logspace(0,np.log(s_max)/np.log(10),self.Ns)
        
        self.grid_s_matrix = np.tile(self.big_S * self.grid_s, (self.ntau,1)).T    #copy s grid to three columns
        
        #iv. relative labor demand (this is pol_n with lowest value normalized to 1)
        self.labor_demand_rel = self.grid_s ** (1 / (1-self.gamma-self.alpha))
        
        #v. approximate the cdf of firms by number of employees
        
        #add zero to front of upper_s
        upper_s_2 = np.insert(df['upper_s'].values, 0, 0)

        for i_idx in range(len(upper_s_2)-1):
            inds = (self.labor_demand_rel > upper_s_2[i_idx]) & (self.labor_demand_rel  <= upper_s_2[i_idx+1]) 
            self.prod_pdf[inds] = df.hs.values[i_idx] / sum(inds) if sum(inds) >0 else 0
            
        self.prod_pdf_matrix = np.tile(self.prod_pdf,(self.ntau,1)).T     #pdf over idiosyncratic draws of s which is copied to three columns
        
        #b. plot data and approximated cdf
        if self.plott:
            plt.scatter(df['upper_s'],df['Hs'],label='Data', c='r')
            plt.plot(self.labor_demand_rel, np.cumsum(self.prod_pdf), label='Model')
            plt.xlabel('Number of Employees (log scale)')
            plt.ylabel('Cummulative Distribution of Establishments')
            plt.xscale('log')
            plt.legend(loc='best')
            plt.title('Distribution of establishments by employment — model vs. data.', fontsize=14)
            plt.savefig('cdf_model_v_data_rr08.pdf')
            plt.show()   
            
            
    

    
    ####################################
    # 2. Helper functions #
    ####################################
    
    def set_tax_system(self, tauv):
        """
        Defines taxes for output, capital and labor

        Parameters
        ----------
        tauv : array of subsidy rate, excempt rate, tax rate

        Returns
        -------
        tau_output, tau_capital, tau_labor : all assigned to tauv depending on policy_type setting

        """
        
        mtau=np.ones((self.Ns,1))*tauv
        
        if self.policy_type == 1 :  #only output tax
            self.tau_output = mtau     
            self.tau_capital = np.zeros((self.Ns, self.ntau))
            self.tau_labor = np.zeros((self.Ns, self.ntau))
        
        elif self.policy_type == 2: #only capital tax
            self.tau_output = np.zeros((self.Ns, self.ntau))      
            self.tau_capital = mtau
            self.tau_labor = np.zeros((self.Ns, self.ntau))
            
        else :  #only labor tax
            self.tau_output = np.zeros((self.Ns, self.ntau))      
            self.tau_capital = np.zeros((self.Ns, self.ntau)) 
            self.tau_labor = mtau
            
    
    
    
    
    
    
    def make_joint_pdf(self, benchmark) :     
        """
        Creates joint pdf of the probability of drawing any combination of tau and s for a firm

        Parameters
        ----------
        benchmark : 1 indicates whether making the joint pdf for the benchmark case. because the benchmark case has no tax/subdidy it is the same as in in the distortion case 1 

        Returns
        -------
        the joint_pdf for each distortion case or benchmark case

        """
        
        #distortion case 1 -- taxes/subsidy uncorrelated with firm size or benchmark case where no tax/subsidy at all
        if self.distortion_case == 1 or benchmark == 1 : 
            self.joint_pdf = self.prod_pdf_matrix * self.policy_pdf    
        
        #distortion case 2 -- tax/subsidy negatively correlated with firm size, subsidize only fraction self.subsidy_frac of lowest prod plants
        if self.distortion_case == 2:
        
          self.joint_pdf = np.zeros((self.Ns,self.ntau))
          prod_cdf = np.cumsum(self.prod_pdf) # cdf over the idiosyncratic draws of s
          I=np.where(prod_cdf <= self.subsidy_frac)
          self.joint_pdf[I,0]=self.prod_pdf[I]     #take the lower part of the pdf over idiosyncratic draws of s
          
          #if there is excempt firms
          if self.excempt_frac>0:
              #take the indices of pdf for s for the interval sub and sub+nosub. 
              I=np.where((prod_cdf > self.subsidy_frac) & (prod_cdf <= self.subsidy_frac + self.excempt_frac))
              self.joint_pdf[I,1] = self.prod_pdf[I]
      
          J=np.where(prod_cdf > self.excempt_frac + self.subsidy_frac)
          self.joint_pdf[J,2]=self.prod_pdf[J]
          
          
        #distortion case 3 -- tax/subsidy positively correlated with firm size, subsidize only fraction self.subsidy_frac of highest prod plants
        elif self.distortion_case == 3:
        
            self.joint_pdf = np.zeros((self.Ns,self.ntau))
            prod_cdf = np.cumsum(self.prod_pdf) # cdf over the idiosyncratic draws of s
            I=np.where(prod_cdf <= 1-self.subsidy_frac - self.excempt_frac)
            self.joint_pdf[I,2]=self.prod_pdf[I]     #take the lower part of the pdf over idiosyncratic draws of s to tax
            
            #if there is excempt firms
            if self.excempt_frac>0:
                #take the indices of pdf for s for the interval sub and sub+nosub. 
                I = np.where((prod_cdf > 1 - self.subsidy_frac - self.excempt_frac) & (prod_cdf <= 1 - self.subsidy_frac))
                self.joint_pdf [I,1] = self.prod_pdf[I]
                
            J=np.where(prod_cdf > 1 - self.subsidy_frac)
            self.joint_pdf[J,0] = self.prod_pdf[J]
            
        
            
            
    
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
   
    
   
    
   
    
   
    
   
    ####################################
    # 3. Problem of the incumbant firm #
    ####################################
    
    def incumbant_firm(self, wage):
        
        """
        The incumbant firms static problem, taking prices and policy as given.
        
        Returns capital and labor demand policy functions. Firm profits and discount PV of incumbant firm.
        """
        
    
        
        # a. demand for capital (capital policy function)
        pol_k = (self.alpha /(self.ret *(1+self.tau_capital)))**((1-self.gamma)/(1-self.gamma-self.alpha)) \
        * (self.gamma /(wage * (1+self.tau_labor)))**(self.gamma/(1-self.gamma-self.alpha)) \
        * (self.grid_s_matrix*(1-self.tau_output))**(1/(1-self.alpha-self.gamma))
        
        # b. demand of labor (labor policy function)
        pol_n = (1+self.tau_capital) * self.ret * self.gamma / ((1+self.tau_labor) * wage * self.alpha) * pol_k
        #pol_n = ((smatrix*(1-self.tau_output) * gamma) / wage)**(1/(1-gamma)) * pol_k**(alpha/(1-gamma))
        
        # c. incumbant profit
        pi=(1-self.tau_output) * self.grid_s_matrix * pol_k**self.alpha * pol_n**self.gamma \
        - (1+self.tau_labor)* wage * pol_n - (1+self.tau_capital) * self.ret * pol_k - self.cf
        
        # d. discounted present value of an incumbent establishment, W(s,pol_k(s,theta))
        W = pi / (1-self.rho)
        
        return pol_k, pol_n, pi, W
    
    
    
    
    
    
    
    
    
    
    
    
    
    ##############################
    # 4. Stationary Equilibrium #
    #############################
    
    
    def entry_condition(self, wage):
        """
        Calculates W_e, which is present discounted value of a potential entrant. In equilibrium it must be zero (W_e = 0 aka the free entry condition). 
        The function takes the wage as given, calculates the incumbant firm's solution and then computes W_e

        Parameters
        ----------
        wage : real wage

        Returns
        -------
        W_e : present discounted value of the entering firm
        pol_k : demand for capital
        pol_n : demand for labor
        pi : profits
        W : present discounted value of the incumbant firm
        pol_enter : policy function for whether to enter or not

        """
         
        #a. solve incumbant firm problem
        pol_k, pol_n, pi, W = self.incumbant_firm(wage) 
        
        # b. enter/stay in the market policy function
        pol_enter=np.zeros((self.Ns,self.ntau))   #1 to enter/stay
        
        for i_s in range(self.Ns):
            for j_t in range(self.ntau):
                if W[i_s,j_t]>=0:
                    pol_enter[i_s,j_t]=1
        
        # c. present discounted value of a potential entrant
        W_e = np.sum(np.sum( W * self.joint_pdf * pol_enter, axis=0) ) - self.ce
        
        return W_e, pol_k, pol_n, pi, W, pol_enter
    
    
    
    
    
    
    
    def find_equilibrium_wage(self, w0_guess, w1_guess):
        """
        Finds the equilibrium wage that satisfies W_e = 0 using the bisection method.

        Parameters
        ----------
        w0_guess : lower bound wage guess
        w1_guess : higher bound wage guess

        Returns
        -------
        w_ss : equilibrium wage

        """

        
        #a. ensure that w0_guess>0 and w1_guess < 0

        assert self.entry_condition(w0_guess)[0] > 0, 'w0, wage lower bound guess too high.'
        
        #i. find a high enough wage such that W_e < 0
        
        for i_w in range(self.maxit):
            We1 = self.entry_condition(w1_guess)[0]
        
            if We1 < 0:
                break
            else: 
                w1_guess += 1
                
        assert self.entry_condition(w1_guess)[0] < 0, 'Can not find upper bound of wage => No convergence'
        
        
        #b. bisection of the wage to find the wage that satisfies W_e = 0
        
        for i_w in range(self.maxit):
            w_guess = (w0_guess + w1_guess)/2
            W_e = self.entry_condition(w_guess)[0]
            
            if np.abs(W_e) < self.tol:
                w_ss = w_guess
                break
            
            else:
                if W_e * We1 > 0 :
                    w1_guess = w_guess
                else :
                    w0_guess = w_guess
                    
        assert i_w+1 < self.maxit, 'find_equilibrium_wage -- No convergence: Zero profit condition not satisfied'
                        
                        
        return w_ss
    
    
    
    
    
    
    def solve_stationary_equilibrium(self) :
        """
        Algorithm to find the stationary equilibrium of the economy. Finds the equilibrium wage, gets the firm's policy functions,
        then calculates the stationary distribution and marginal distributions and finally the aggregate statistics of the economy.

        Returns
        -------
        Y_ss : steady state output
        K_ss : steady state capital
        TFP_ss : steady state TFP
        average_firm_size : average number of employees employeed by a firm
        E_star : steady state entry mass
        Y_set : percentage of output produced that recieves a subsidy, is excempt or is taxed. adds up to 1.
        subsidy_size : the size of the subsidy relative to ss output
        N_ss : steady state labor demanded
        w_ss : steady state wage
        cdf_stationary : marginal productivity cdf
        cdf_emp : cdf of employment by firm productivity
        """
        
        
        
        
        #a. find the equilibrium wage given the tax rate and subsidy
        w_ss = self.find_equilibrium_wage(self.w0_guess, self.w1_guess)
    
        #b. obtain firm policy functions and discount present value factors
        W_e , pol_k, pol_n, pi, W, pol_enter = self.entry_condition(w_ss)
        
        
        #c. obtain the invariant distribution 
        
        #i. normalized invariant distribution over firms
        mu_hat = pol_enter/self.lambdaa * self.joint_pdf
    
        #ii. labor market clearing (section 3.5), agg demand for labor
        N_ss = np.sum(np.sum(pol_n*mu_hat, axis=0))
        
        #iii. ss equilibrium level of entry (mass of entrants)
        E_star = 1/N_ss 
        
        #iv. rescale invariant distribution over firms, mu(s,tau)
        mu = E_star*mu_hat
        
        #d. marginal distributions
        
        #i. sum over subsidies, except, taxes of stationary distribution
        distrib_stationary = np.sum(mu, axis=1)
        total_mass = np.sum(distrib_stationary)
        
        #ii. marginal stationary distribution over productivity
        pdf_stationary = distrib_stationary / total_mass
        cdf_stationary = np.cumsum(pdf_stationary)
        
        #iii. stationary distribution over number of employed 
        distrib_emp = (pol_n[:,2] * pdf_stationary)/ np.sum(pol_n[:,2] * pdf_stationary)
        pdf_emp = distrib_emp / np.sum(distrib_emp)
        cdf_emp = np.cumsum(pdf_emp)
        
        #e. Aggregate statistics
        
        Y_ss = np.sum(np.sum( self.grid_s_matrix * pol_k**self.alpha * pol_n**self.gamma*mu, axis=0))   #ss output
        K_ss = np.sum(np.sum(pol_k*mu, axis=0))     #ss capital
        TFP_ss = Y_ss/(N_ss*E_star)/(K_ss/(N_ss*E_star))**self.alpha
        total_employment = np.dot(self.labor_demand_rel, distrib_stationary)
        average_firm_size = total_employment / total_mass
        
        #output share of subsidy, excemption, taxed
        Y_set = np.sum(self.grid_s_matrix * pol_k**self.alpha*pol_n**self.gamma*mu, axis=0) / Y_ss
        
        Y_sub_percent = Y_set[0]    #output share of establishments that are receiving a subsidy, Y_s/Y
        Y_exempt_percent = Y_set[1]
        Y_taxed__Percent = Y_set[2]
       
        #the total subsidies paid out to establishments receiving subsidies as a fraction of output. numerator takes first column which is subsidy (S/Y)
        subsidy_size = np.sum(-self.tau_output[:,0]*self.grid_s_matrix[:,0]*pol_k[:,0]**self.alpha \
                              *pol_n[:,0]**self.gamma*mu[:,0]-self.tau_capital[:,0]*self.ret \
                                  *pol_k[:,0]*mu[:,0]-self.tau_labor[:,0]*w_ss* \
                                      pol_n[:,0]*mu[:,0]) / Y_ss
        
    
        return Y_ss, K_ss, TFP_ss, average_firm_size, E_star, Y_set, subsidy_size, N_ss, w_ss, cdf_stationary, cdf_emp
    
    
    
    
    
    
    
    
    
    
    ############################
    # 5. Distortionary Policy #
    ###########################
    
    
    def find_subsidy_rate(self, tau):
        """
        Given a tax rate and excemption rate, finds the subsidy rate that will generate the same steady state capital in the distorted
        economy as in the benchmark equilibrium. 

        Parameters
        ----------
        tau : tax rate

        Returns
        -------
        taus_star : subsidy rate that will generate the same capital stock as the benchmark economy

        """
        
        #a. find residual with lower and upper guesses
        tauv_0 = np.array([-self.tau_s_0, 0, tau])    #subsidy rate, excempt rate, tax rate  
        self.set_tax_system(tauv_0) 
        
        Kss_d = self.solve_stationary_equilibrium()[1]
        residual0 = Kss_d / self.Kss_b - 1
        
        tauv_1 = np.array([-self.tau_s_1, 0, tau])    #subsidy rate, excempt rate, tax rate  
        self.set_tax_system(tauv_1) 
        
        Kss_d = self.solve_stationary_equilibrium()[1]
        residual1 = Kss_d / self.Kss_b - 1
        
        assert residual0*residual1 < 0, 'find_subsidy_rate -- WARNING: No equilibrium tau exists'
        
        #b. bisection to find the subsidy rate that genereates the same ss capital as in benchmark case
        
        tau_s_0 = self.tau_s_0 
        tau_s_1 = self.tau_s_1
        
        for i_t in range(self.maxit):
            taus = (tau_s_0 + tau_s_1)/2
            tauv = np.array([-taus, 0, tau])    #subsidy rate, excempt rate, tax rate  
            self.set_tax_system(tauv) 
            
            Kss_d = self.solve_stationary_equilibrium()[1]
            residual = Kss_d / self.Kss_b - 1
            
            if np.abs(residual) < self.tol:
                
                taus_star = taus
                break
            
            else:
                if residual1 * residual>0 :
                    tau_s_1 = taus
                else :
                    tau_s_0 = taus
        
        assert i_t+1 < self.maxit, 'find_subsidy_rate -- taus has not converged'
                
        return taus_star
        
    
    
    
    
    
    
    #####################
    # 6. Main function #
    ####################
    
    
    def solve_model(self):
        """
        Finds the stationary equilibrium in the benchmark economy, then finds the stationary equlibrium(s) in 
        the distorted economies and compares steady state aggregate statistics relative to the benchmark statistics.
        """ 
        
        t0 = time.time()    #start the clock
        
        # a. benchmark case
        
        #i. joint pdf of productivity state and tau 
        self.make_joint_pdf(1)
        
        #ii. set policy. in RR08 the benchmark economy has no taxes nor subsidies
        self.tau_benchmark = np.array([0, 0, 0])    #subsidy rate, excempt rate, tax rate  
        self.set_tax_system(self.tau_benchmark)          #set tax system
        
        #iii. benchmark equilibrium
        self.Yss_b, self.Kss_b, self.TFPss_b, self.average_firm_size_b, self.E_star_b, _, \
            _, self.N_ss_b, self.w_ss_b, self.cdf_stationary_b, self.cdf_emp_b = self.solve_stationary_equilibrium()
        
        print("\n-----------------------------------------")
        print("Benchmark Stationary Equilibrium")
        print("-----------------------------------------")
        print(f"ss output  = {self.Yss_b:.2f}")
        print(f"ss capital  = {self.Kss_b:.2f}")
        print(f"ss tfp  = {self.TFPss_b:.2f}")
        print(f"ss wage  = {self.w_ss_b:.2f}")
        print(f"entry mass = {self.E_star_b:.3f}")
        print(f"avg. firm size = {self.average_firm_size_b:.2f}")
        
        #b. plot (note that the distributions plotted here are unaffected by the distortionary policies)
        
        if self.plott:
            #i. initialize
            employed = [4.99, 49.99]
            firm_size_by_employee  = np.zeros(len(employed)+1)
            share_employment = np.zeros(len(employed)+1)
            
            
            #i. percentage of firms that employ employed
            
            for i_e in range(len(employed)):
                summ = np.sum(firm_size_by_employee)
                interpolate = self.interpol(self.labor_demand_rel, self.cdf_stationary_b, employed[i_e])[0] #labor_demand_rel is labor demand with the lowest value normalized to 1
                firm_size_by_employee[i_e] = interpolate - summ
            firm_size_by_employee[-1] = 1 - np.sum(firm_size_by_employee)
            
            plt.pie(firm_size_by_employee, labels=['<5','5<50','50 =<'], autopct="%.1f%%")
            plt.title('Size of Firms by Firm Size (Number of Employees)')
            plt.savefig('firm_size_rr08.pdf')
            plt.show()
            
            
            #ii. employment percentage by firm size
            for i_e in range(len(employed)):
                summ = np.sum(share_employment)
                interpolate = self.interpol(self.labor_demand_rel, self.cdf_emp_b , employed[i_e])[0]
                share_employment[i_e] = interpolate - summ
            share_employment[-1] = 1 - np.sum(share_employment)
            
            plt.pie(share_employment, labels=['<5','5<50','50 =<'], autopct="%.1f%%")
            plt.title('Employment Share by Firm Size (Number of Employees)')
            plt.savefig('employment_by_firm_size_rr08.pdf')
            plt.show()
            
            #iii. productivity cdf and employment cdf
            plt.plot(self.grid_s, self.cdf_stationary_b)
            plt.plot(self.grid_s, self.cdf_emp_b)
            plt.title('Stationary CDF' )
            plt.xlabel('Productivity level')
            plt.ylabel('Cumulative Sum')
            plt.legend(['Firms by Productivity Level','Share of Employment'])
            plt.savefig('cdf_rr08.pdf')
            plt.show()
        
        
        
        #c. distortion case
        
        #i. joint pdf of productivity state and tau 
        self.make_joint_pdf(0)
        
        #ii. compute stationary economy for each tau
        
        for idx, tau in enumerate(self.tau_vector):
            
            #iii. find the subsidy rate that generates the same capital stock as in benchmark economy
            self.tau_s[idx] = self.find_subsidy_rate(tau)
            
            # set tax system with newly found tau_s and given tau
            tauv = np.array([-self.tau_s[idx], self.excempt_frac, tau])    #subsidy rate, excempt rate, tax rate  
            self.set_tax_system(tauv)          #set tax system
            
            #v. distorted stationary equilibrium
            self.Yss_d[idx], self.Kss_d[idx], self.TFPss_d[idx], self.average_firm_size_d[idx], self.E_star_d[idx], \
            self.Y_set_d[idx,:], self.subsidy_size_d[idx], self.N_ss_d[idx], self.w_ss_d[idx],\
            _, _  = self.solve_stationary_equilibrium()
            
        print("\n-----------------------------------------")
        print("Distorted Stationary Equilibrium")
        print("-----------------------------------------\n")
        if self.distortion_case == 1:
            print("Tax/Subidy Uncorrelated with Firm Level Producitivity\n")
        elif self.distortion_case == 2:
            print("Tax/Subidy Negatively Correlated with Firm Level Producitivity")
            print("(low productivity firms recieve subsidy, high productivity taxed)\n")
        elif self.distortion_case == 2:
            print("Tax/Subidy Positively Correlated with Firm Level Producitivity")
            print("(high productivity firms recieve subsidy, low productivity taxed)\n")
        if self.policy_type == 1 :
            print("Tax Type: Tax on output\n")
        elif self.policy_type == 2 :
            print("Tax Type: Tax on capital\n")
        elif self.policy_type == 3 :
            print("Tax Type: Tax on labor\n")
        print(f"fraction of firms recieving subsidy = {self.subsidy_frac:.2f}")
        print(f"fraction of firms taxed = {1-self.subsidy_frac-self.excempt_frac:.2f}")
        print(f"fraction of firms excempt = {self.excempt_frac:.2f}")
        print("-----------------------------------------\n")
        
        print(tabulate([['relative Yss', round(self.Yss_d[0]/self.Yss_b, 2), round(self.Yss_d[1]/self.Yss_b, 2), round(self.Yss_d[2]/self.Yss_b, 2), round(self.Yss_d[3]/self.Yss_b, 2)],
                        ['relative TFPss', round(self.TFPss_d[0]/self.TFPss_b, 2), round(self.TFPss_d[1]/self.TFPss_b, 2), round(self.TFPss_d[2]/self.TFPss_b, 2), round(self.TFPss_d[3]/self.TFPss_b, 2)], 
                       ['relative entry mass', round(self.E_star_d[0]/self.E_star_b, 2), round(self.E_star_d[1]/self.E_star_b, 2), round(self.E_star_d[2]/self.E_star_b, 2), round(self.E_star_d[3]/self.E_star_b, 2)],
                       ['share of subsidized output', round(self.Y_set_d[0,0], 2), round(self.Y_set_d[1,0], 2), round(self.Y_set_d[2,0], 2), round(self.Y_set_d[3,0], 2)],
                       ['total subsidy paid of output', round(self.subsidy_size_d[0], 2), round(self.subsidy_size_d[1], 2), round(self.subsidy_size_d[2], 2), round(self.subsidy_size_d[3], 2)],
                       ['subsidy rate (tau_s)', round(self.tau_s[0], 2), round(self.tau_s[1], 2), round(self.tau_s[2], 2), round(self.tau_s[3], 2)],
                       [], 
                       ['relative Kss', round(self.Kss_d[0]/self.Kss_b, 2), round(self.Kss_d[1]/self.Kss_b, 2), round(self.Kss_d[2]/self.Kss_b, 2), round(self.Kss_d[3]/self.Kss_b, 2)], 
                       ['relative wss', round(self.w_ss_d[0]/self.w_ss_b, 2), round(self.w_ss_d[1]/self.w_ss_b, 2), round(self.w_ss_d[2]/self.w_ss_b, 2), round(self.w_ss_d[3]/self.w_ss_b, 2)], 
                       ['relative Nss', round(self.N_ss_d[0]/self.N_ss_b, 2), round(self.N_ss_d[1]/self.N_ss_b, 2), round(self.N_ss_d[2]/self.N_ss_b, 2), round(self.N_ss_d[3]/self.N_ss_b, 2)], 
                       ['relative avg. firm size', round(self.average_firm_size_d[0]/self.average_firm_size_b, 2), round(self.average_firm_size_d[1]/self.average_firm_size_b, 2), round(self.average_firm_size_d[2]/self.average_firm_size_b, 2), round(self.average_firm_size_d[3]/self.average_firm_size_b, 2)]],
                       headers=['Variable', 'Tax = '+str(self.tau_vector[0]), "Tax = "+str(self.tau_vector[1]), 'Tax = '+str(self.tau_vector[2]), 'Tax = '+str(self.tau_vector[3])]))
        

        t1 = time.time()
        print(f'\nTotal Run Time: {t1-t0:.2f} seconds')
#run everything

#distortion case 1 -- tax/subsidy uncorrelated with firm size
rr08=RestucciaRogerson(distortion_case = 1)
rr08.solve_model()

#distortion case 2 -- tax/subsidy negatively correlated with firm size. only small firms recieve subsidy and large ones taxed
rr08=RestucciaRogerson(distortion_case = 2)
rr08.solve_model()

