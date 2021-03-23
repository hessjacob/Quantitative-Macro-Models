# Quantitative-Macro-Models
This is a collection of code for quantitative macroeconomic models that I have written as personal learning exercises. Mostly all the codes have heterogenous agents and are written in python using numba. References used can be found in each file.    

**Quick Guide**
- Heterogenous households
  * Aiyagari
  * Consumption Saving
 
- Heterogenous firms
  * Hopenhayn
  * Restuccia and Rogerson
 
- Representative household
  * Neoclassical growth
  * RANK models

# Aiyagari 
Stationary equilibrium solution in a production economy with incomplete markets and no aggregate uncertainty. Heterogenous agents are infinitely lived and are exposed to idiosyncratic income risk. The versions differ in how the household problem is solved and how the income shock process is specified.

**Shared features in all versions is** 

1) Plots the wealth distribution, capital supply and demand and policy functions.
2) Exogenous borrowing constraint which the user can choose. 
3) Unless otherwise specified, a monte carlo simulation is used to approximate the stationary distribution

**Codes and Solution Methods**

- Value Function Iteration
  * Version 1 -- 2 income states. 
  * Version 2 -- Code will replicate Aiyagari (1994). A slight difference is that the continuous income process which is discretely approximated up to seven different income states using the Rouwenhorst method rather than Tauchen. The Tauchen method is available in the code should the user want an exact replication. 
  
- Endogenous Grid Method
  * Version 1 -- 2 income states. 
  * Version 2 -- Code will replicate Aiyagari (1994). A slight difference is that the continuous income process which is discretely approximated up to seven different income states using the Rouwenhorst method rather than Tauchen. The Tauchen method is available in the code should the user want an exact replication. 


# Consumption Saving in Incomplete Markets (aka the income flucuation problem)
Partial equilibrium solution (prices are exogenously set) for heterogenous agents that are infinitely lived in incomplete markets and are exposed idiosyncratic income risk. The versions differ in how the household problem is solved, how the income shock process is specified and how the stationary distribution is approximated. These codes are extended to solve for general equilibrium in the Aiyagari section. 

**Shared features in all versions is** 

1) Runs a markov chain simulation for 50,000 heterogenous households and the stationary distribution is approximated. 
2) Exogenous borrowing constraint which the user can choose. 

**Codes and Solution Methods**

- Value Function Iteration
  * Version 1 -- 2 income states. 
  * Version 2 -- Continuous income process which is discretely approximated up to seven different income states using the Rouwenhorst method. 
  
- Endogenous Grid Method
  * Version 1 -- 2 income states. 
  * Version 2 -- Continuous income process which is discretely approximated up to seven different income states using the Rouwenhorst method. 
  * Version 3 -- Based on Alisdair McKay's method (https://alisdairmckay.com/Notes/HetAgents/EGM.html) which is a variant of the prior versions. In addition to approximating the stationary density via monte carlo it also solves for it using an eigenvalue method and plots the comparision of the densities. 
  
# Hopenhayn
Finds the stationary equilibrium in a dynamic model with heterogenous firms exposed to idiosyncratic productivity levels, no aggregate uncertainty and endogenous entry/exit of firms as in Hopenhayn (1992). It is a partial equilibrium solution as the demand side of the economy is exogenously given and wages are normalized to one. Hopenhayn and Rogerson (1993) extended this to general equilibrium. Here we have an industry with many firms that is competitive and produces a single homoegenous good. Every period incumbent firms choose whether to exit the market. There is free entry into the industry, subject to paying a fixed entry cost. The equilibrium of the industry determines the price and quantity of the good and the amount of labor hired in the industry. The code is solved using value function iteration to solve the firm problem and analytically solves for the stationary distribution. 

# Neoclassical Growth (Deterministic and Stochastic)
- Social planner solution for complete markets. 
- VFI to solve the model and Chebyshev polynomial approximation of decision rules to simulate.
- Computes the solution and does a simulation.

# Representative Agent New Keynesian (RANK)
Under the assumption complete markets these models solve for a representative agent with nominal frictions. Both models include a version with an occassionally binding constraint on the nominal interest rate. 

**System requirements**
- Dynare, which is an opensource software (download at https://www.dynare.org/download/) which is run through MATLAB. I used version 4.5.7. 
- For the ZLB, Occbin toolbox (download at https://www.matteoiacoviello.com/research.htm). I used version occbin_20140630. 

**Codes and Solution Methods**
- New Keynesian 
  * Standard New Keynesian model as laid out in Chapter 3 in "Monetary Policy, Inflation, and the Business Cycle" by Jordi Galí
  * Calvo price frictions 
  
- DSGE 
  * Based on Christiano et. al. (2005) which adds more nominal frictions and shocks to better replicate macro data. 
  * My version is calibrated to match data moments but it can easily be estimated using bayesian techniques. 
  * Macro data is cleaned and included


# Restuccia and Rogerson
Replicates Restuccia and Rogerson 2008 which show that resource misallocation across heterogenous firms can have sizeable negative effects on aggregate output and TFP even when policy does not relay on aggregate capital accumulation or aggregate relative price differences. The paper highlights the importance of resource misallocation across firms with different levels or productivity and could potentially explain cross-country differences in output per capita. The code calculates the efficient or benchmark economy and then compares  economies under a policy distortion of either output, capital or labor which reallocates resources among firms through tax/subsidies. Each firm faces its own tax or subdidy. To emphasize the effects, for each tax rate the code finds the subsidy rate that will generate the same aggregate capital stock as the benchmark economy. The focus is on policies that create idiosyncratic distortions to establishment-level decisions and hence cause a reallocation of resources across establishments. 
