"""
Author: Jacob Hess 
Date: December 2020

Written in python 3.8 on Spyder IDE.

Description: This code solves the consumption/saving problem (aka the income flucuation problem) for the infinitely 
household in partial equilibrium using the endogenous grid method. In addition, it runs a simulation, computes the 
invariant distribution and calculates the euler equation error over the simulation.

Aknowledgements: I used notes and pieces of code from the following :
    1) Alisdair McKay https://alisdairmckay.com/Notes/HetAgents/EGM.html
    2) Jeppe Druedahl (https://github.com/JeppeDruedahl) and NumEconCopenhagen (https://github.com/NumEconCopenhagen)
    
Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual)
    -- Interpolation from EconForge
       * optimized interpolation routines for python/numba
       * to install 'conda install -c conda-forge interpolation'
       * https://github.com/EconForge/interpolation.py
"""



import time
import numpy as np
from numba import njit, prange
from interpolation import interp
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')





#############
# I. Model  #
############

class ConSaveEGM:

    """
    Class object of the model. The command ConSaveEGM.solve_model() runs everything.
    """    

    ############
    # 1. setup #
    ############

    def __init__(self, sigma=2,               #crra coefficient
                       a_bar = 0,             #select borrowing limit
                       plott =1               #select 1 to make plots
                       ):
        
        #parameters subject to changes
        self.sigma, self.a_bar, self.plott =sigma, a_bar, plott

        self.setup_parameters()
        self.setup_grid()
        
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
        self.tol_hh = 1e-15  # tolerance for policy functions iterations
        self.maxit = 2000  # maximum number of policy functions iterations

        # income
        self.Nz = 2
        self.grid_z = np.array([0.5, 1.5])                #productuvity states
        self.pi = np.array([[3/4, 1/4],[1/4, 3/4]])   #transition probabilities
        

        # savings grid
        # 1) number of of grid points: Ideally have a low number like 30-50 when using exponential grid and EGM.
        # 2) if you need many grid points (100+) an exponential grid could return unstable solutions using EGM. 
        #    linspace instead will always give stable solutiobs.
        
        self.Ns = 50
        self.sav_min = self.a_bar
        self.sav_max = 20
        self.curv = 3
        
        # c. simulation
        self.seed = 123
        self.a0 = 1.0  # initial cash-on-hand (homogenous)
        self.simN = 50_000  # number of households
        self.simT =  500 # number of time periods to simulate

        


    def setup_grid(self):

        # a. savings (or end-of-period assets) grid
        self.grid_sav = self.make_grid(self.sav_min, self.sav_max, self.Ns, self.curv)  

        # b. initial distribution of z
        z_diag = np.diag(self.pi ** 1000)
        self.ini_p_z = z_diag / np.sum(z_diag)

        avg_z = np.sum(self.grid_z * self.ini_p_z)
        self.grid_z = self.grid_z / avg_z  # force mean one
        




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
     
    def solve_egm(self, G):
        
        """
        Endogenous grid method.
        
        *Input
            - G : Savings policy function
        *Output
            - a : (a,z) mapping to grid_sav
            - c : consumption policy function
        """
        
        # a. Calculate the RHS of the Euler Equation 
        c_prime=np.empty((self.Nz,self.Ns))
        avg_marg_u_prime = np.zeros((self.Nz, self.Ns))
        
        # i. Use the budget constraint to get c'. Recall that G(a,z) and interpolated last term is a'' = G(G(a,z),z')
        for iz in range(self.Nz) :
            c_prime[iz] = (1+self.ret) * self.grid_sav + self.w*self.grid_z[iz] - self.interpol(G[iz,:],self.grid_sav,self.grid_sav)[0] 
        
        # ii. Marginal utility u'(c')
        marg_u_prime = self.u_prime(c_prime)
        
        # iii. Calculate expectation
        for iz in range(self.Nz):
            avg_marg_u_prime_izz = 0
            
            for izz in range(self.Nz):
                avg_marg_u_prime_izz += self.pi[izz,iz] * marg_u_prime[izz]
                
            avg_marg_u_prime[iz,:] = avg_marg_u_prime_izz
            
        # b. Solve for c
        # i. Use RHS of EE to get u'(c) on LHS
        marg_c = self.beta*(1+self.ret)*avg_marg_u_prime

        # ii. Invert u'(.) to solve for c
        c = self.u_prime_inv(marg_c)
        
        # c. Obtain endogenously determined asset grid
        a = np.empty((self.Nz,self.Ns))
        
        # i. use budget constraint to find assets today. a = (a' + c - w*z)/(1+r).
        # solving for a here means we have a now have mapping of (a,z) to a'. This will be the new policy function guess. 
        for iz in range(self.Nz):
            a[iz,:] = (self.grid_sav + c[iz,:] - self.w*self.grid_z[iz])/ (1+self.ret)
        
        # ii. ensure that assets satisfy borrowing constraint
        threshold = a < self.a_bar
        a[threshold] = self.a_bar
            
        
        return a, c
           
    
    def solve_hh(self) :
        
        """
        Solves the household problem.
        
        *Output
            - pol_sav : Savings policy function
            - pol_cons : Consumption policy function
        """
        
        print('\nSolving household problem...')
        
        # a. initialize policy rule G(a,z). grid_sav is a'.
        G = 10+0.1*self.grid_sav*np.ones((self.Nz,self.Ns)) 
        
        # b. iteration
        for it_hh in range(self.maxit):
            a, pol_cons = self.solve_egm(G)
            
            # i. take the difference of the new and old policy functions
            diff_hh = np.linalg.norm(a-G)
            
            if diff_hh  < self.tol_hh:
                print(f"Convergence in {it_hh} iterations")
                break
            
            # ii. set the mapping (a,z) to a' equal to G(a,z) for next iteration
            G = a
            
        if it_hh > self.maxit:
            print('No convergence')
            
        # c. get savings policy function
        pol_sav = np.empty((self.Nz,self.Ns))
        
        for iz in range (self.Nz):
            pol_sav[iz,:] = self.interpol(a[iz,:],self.grid_sav,self.grid_sav)[0]
        
        return pol_sav, pol_cons, G
    
    
    

    ############################
    # 5.Invariant Distribution #
    ############################
    
    def MakeTransMat(self,G):
        
        T = np.zeros((self.Nz*self.Ns, self.Nz*self.Ns))
        for j in range(self.Nz):
            x, i = self.interpol(G[j],self.grid_sav,self.grid_sav)
            p = (self.grid_sav-G[j,i-1]) / (G[j,i] - G[j,i-1])
            p = np.minimum(np.maximum(p,0.0),1.0)
            sj = j*self.Ns
            for k in range(self.Nz):
                sk = k * self.Ns
                T[sk + i,sj+np.arange(self.Ns)]= p * self.pi[k,j]
                T[sk + i - 1,sj+np.arange(self.Ns)] = (1.0-p)* self.pi[k,j]
    
        assert np.allclose(T.sum(axis=0), np.ones(self.Nz*self.Ns))
        
        return T
    
    def GetStationaryDist(self,T):
        eval,evec = np.linalg.eig(T)
        i = np.argmin(np.abs(eval-1.0))
        D = np.array(evec[:,i]).flatten()
        assert np.max(np.abs(np.imag(D))) < 1e-6
        D = np.real(D)  # just recasts as float
        return D/D.sum()
        

    
    
    
    #####################
    # 6. Main Function  #
    #####################


    def solve_model(self):
    
        """
        Runs the entire model.
        """    
        
        t0 = time.time()    #start the clock
        
        # a. solve household problem 
        
        self.pol_sav, self.pol_cons, G = self.solve_hh()
        
        t1 = time.time()
        print(f'Household problem time elapsed: {t1-t0:.2f} seconds')
         
        # b. simulation
        
        print("\nSimulating...")
        
        # i. initial values for agents
        a0 = self.a0 * np.ones(self.simN)
        z0 = np.zeros(self.simN, dtype=np.int32)
        z0[np.linspace(0, 1, self.simN) > self.ini_p_z[0]] = 1
        
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
            self.seed,
        )
        
        t2 = time.time()
        print(f'Simulation time elapsed: {t2-t1:.2f} seconds')
        
        
        #c. Calculate the invariant distribution
        print("\nFinding invariant distribution...")
        self.Distribution = self.GetStationaryDist(self.MakeTransMat(G))
        
        # i. aggregate asset holdings
        self.agg_asset = (self.grid_sav.reshape(1,self.Ns) * self.Distribution.reshape(self.Nz,self.Ns)).sum()
        
        # ii. density 
        normfactor = np.hstack((1.0,np.diff(self.grid_sav)))[np.newaxis,:]
        self.Density = self.Distribution.reshape(self.Nz,self.Ns)/normfactor
        
        # iii. marginal wealth density
        self.wealth_density = (self.Density.sum(axis=0))/self.Nz
        
        t3 = time.time()
        print(f'Distribution calculation time elapsed: {t3-t2:.2f} seconds')
        
        
        # c. plot
        
        if self.plott:
            
            print('\nPlotting...')
            
            ##### Policy Functions #####
            plt.plot(self.grid_sav, self.pol_sav.T)   
            plt.plot(self.grid_sav,self.grid_sav,':')
            plt.title('Savings Policy Function')
            plt.xlabel('Current Assets')
            plt.ylabel('Savings')
            plt.legend(['z='+str(self.grid_z[0]),'z='+str(self.grid_z[1]),'45 degree line'])
            plt.savefig('savings_policyfunction_egm.pdf')
            plt.show()
            
            plt.plot(self.grid_sav, self.pol_cons.T)
            plt.title('Consumption Policy Function')
            plt.xlabel('Current Assets')
            plt.ylabel('Consumption')
            plt.legend(['z='+str(self.grid_z[0]),'z='+str(self.grid_z[1])])
            plt.savefig('consumption_policyfunction_egm.pdf')
            plt.show()
    
            
            ##### Simulation ####
            fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,6))
            fig.tight_layout(pad=4)
            
            #first individual over first 100 periods
            ax1.plot(np.arange(0,99,1), self.sim_sav[:99,1], np.arange(0,99,1), self.sim_c[:99,1],
                     np.arange(0,99,1), self.grid_z[self.sim_z[:99,1]],'--')
            ax1.legend(['Savings', 'Consumption', 'Income'])  
            ax1.set_title('Simulation of First Household During First 100 Periods')
            
            #averages over entire simulation
            ax2.plot(np.arange(0,self.simT,1), np.mean(self.sim_sav, axis=1), 
                     np.arange(0,self.simT,1), np.mean(self.sim_c, axis=1) )
            ax2.legend(['Savings', 'Consumption', 'Income'])
            ax2.set_title('Simulation Average over 50,000 Households')
            plt.savefig('simulation_egm.pdf')
            plt.show()
            
            
            ##### Distributions ####
            
            plt.plot(self.grid_sav,self.Density.T)
            plt.title('Joint Density')
            plt.xlabel('Current Assets')
            plt.ylabel('Density')
            plt.legend(['z='+str(self.grid_z[0]),'z='+str(self.grid_z[1])])
            #plt.savefig('joint_density_egm.pdf')
            plt.show()
            
            plt.plot(self.grid_sav,self.wealth_density.T,'--',color='tab:red')
            sns.histplot(self.sim_sav[-1,:], bins=100, stat='density')
            plt.title('Marginal Wealth Density Comparision')
            plt.xlabel('Current Assets')
            plt.ylabel('Density')
            plt.legend(['Eigenvector Method','Monte Carlo Simulation'])
            plt.savefig('wealth_density_egm.pdf')
            plt.show()
            
            t4 = time.time()
            print(f'Plot time elapsed: {t4-t3:.2f} seconds')
            

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
        
        
        t5 = time.time()
        print(f'\nTotal Run Time: {t5-t0:.2f} seconds')




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
    seed,
        ):
    
    """
    Simulates markov chain for T periods for N households. Also checks 
    the grid size by ensuring that no more than 1% of households are at
    the maximum value of the grid.
    
    *Output
        - sim_c: consumption profile
        - sim_sav: savings (a') profile
        - sim_z: income profile index, 0 for low state, 1 for high state
        - sim_m: cash-on-hand profile ((1+r)a + w*z)
        - euler_error : error when the euler equation equality holds
    """
    
    
    # 1. initialization
    np.random.seed(seed)
    
    sim_sav = np.zeros((simT,simN))
    sim_c = np.zeros((simT,simN))
    sim_m = np.zeros((simT,simN))
    sim_z = np.zeros((simT,simN), np.int32)
    sim_z_idx = np.zeros((simT,simN), np.int32)
    edge = 0
    euler_error = np.empty((simT,simN)) * np.nan
    
    
    
    
    # 2. helper functions
    
    # savings policy function interpolant
    polsav_interp = lambda a, z: interp(grid_sav, pol_sav[z, :], a)
    
    #marginal utility
    u_prime = lambda c : c**(-sigma)
    
    #inverse marginal utility
    u_prime_inv = lambda x : x ** (-1/sigma)
    
    
    
    # 3. simulate markov chain
    for t in range(simT):   #time

        draw = np.linspace(0, 1, simN)
        np.random.shuffle(draw)
        
        for i in prange(simN):  #individual hh

            # a. states 
            if t == 0:
                z_lag = np.int32(z0[i])
                a_lag = a0[i]
            else:
                z_lag = sim_z[t-1,i]
                a_lag = sim_sav[t-1,i]
                
            # b. shock realization. 0 for low state. 1 for high state.
            if draw[i] <= pi[z_lag, 1]:     #state transition condition
                sim_z[t,i] = 1
                sim_z_idx[t,i] = 0  #state row index
            else:
                sim_z[t,i] = 0
                sim_z_idx[t,i] = 1
                
            # c. income
            y = sim_w*grid_z[sim_z[t,i]]
            
            # d. cash-on-hand path
            sim_m[t, i] = (1 + sim_ret) * a_lag + y
            
            # e. consumption path
            sim_c[t,i] = sim_m[t, i] - polsav_interp(a_lag,sim_z[t,i])
            
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



cs_EGM=ConSaveEGM()
cs_EGM.solve_model()


