# Quantitative-Macro-Models
This is a collection of code for quantitative macroeconomic models that I have written as personal learning exercises. All the codes, aside from the DSGE block, have heterogenous agents and are written in python using numba. You can find different versions under each topic where I might use different computational methods, increase the scale of the model and/or adjust some key assumptions. I have worked hard to make them readable and very fast so that they might help others who are interested in learning about these topics. References used can be found in each file.    

**Quick Guide**
- Heterogenous households
  * Aiyagari
  * Consumption Saving
 
- Heterogenous firms/Industry dynamics
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
  
# Hopenhayn -- Firm Dynamics
**Version 1 -- Hopenhayn (1992)**
- Finds the stationary equilibrium in a dynamic model with heterogenous firms exposed to idiosyncratic productivity levels, no aggregate uncertainty and endogenous entry/exit of firms as in Hopenhayn (1992). It is a partial equilibrium solution as the demand side of the economy is exogenously given and wages are normalized to one.

*Detailed Description*.
The economy consists of many firms that are competitive and produce a single homoegenous good. Every period incumbent firms choose whether to exit the market. There is free entry into the industry, subject to paying a fixed entry cost. The stationary equilibrium determines the price and quantity of the good and the amount of labor hired. The code is solved using value function iteration to solve the firm problem and analytically solves for the stationary distribution. 

This model lays the basis for heterogenous firm/industry dynamics models. Hopenhayn and Rogerson (1993) extend this to general equilibrium. In version 2 I extend this to general equilibrium by embedding the neoclassical growth model solving for the stationary equilibrium.

**Version 2 -- Firm Dynamics (Hopenhayn 1992) and the Neoclassical Growth Model (household owns capital)**
   
* The code embeds the standard neoclassical growth model into Hopenhayn (1992) and solves for the stationary equilibrium in which there is continuous 
entry/exit of firms. The model written here is a loose variant of Veracierto (2001) who was the first (to my knowledge) to write a neoclassical growth model with 
firm dynamics. I solve the household problem using value function iteration and approximate the stationary productivity distribution of firms by fixed point iteration.

* This is a frictionless economy.

*Detailed Description*.
The difference between this model and Hopenhayn (1992) and its general equilibrium extension in Hopenhayn and Rogerson (1993) is:

1) Unlike in either model, there is a flexible form of capital that the firm is able to rent from households (therefore they make the investment decision) as is standard in the neoclassical growth model. 
2) Labor is inelastically supplied by the household and not divisble like in Hopenhayn and Rogerson (1993).

Agents are infinitely lived and ex-ante identical. There are complete markets which allows me to to construct a repersentative household. Firms, on the other hand, are heterogenous in their productivity.The economy considered is similar to the neoclassical growth model except for output, which is produced by a large number of establishments subject to idiosyncratic productivity shocks that induce them to expand and contract over time.There is no aggregate uncertainty.Establishments have access to a decreasing returns to scale technology, pay a one-time fixed cost of entry, and a fixed cost of operation every period. 

There is ongoing exogenous and endogenous entry/exit in the steady state. Firms may exogenously die with probability lambda every period. I include this so that there are large productive firms that might suddenly shut down. The timing is as follows. At the beginning of every period firms that receive an exit shock leave the market. Remaining firms draw their new productivity and endogenously decide whether to continue or shut down. Those that continue choose the capital and labor factor demands to maximize profits. Meanwhile, there is a mass of potential entrants who draw an intial productivity level and decide whether they should enter the market. Before entering they must pay a one-time fixed entry cost.  

**Version 3 -- Firm Dynamics (Hopenhayn 1992) and the Neoclassical Growth Model (firm owns capital)**

* The code embeds the standard neoclassical growth model into Hopenhayn (1992) and solves for the stationary equilibrium in which there is continuous 
entry/exit of firms. The difference with this and version 2 is that the firm owns the capital stock and makes the investment decision. The model is a simplified general equilbrium extension of Clementi and Palazzo (2019). I solve the household problem using value function iteration and approximate the stationary joint distribution of firms by fixed point iteration on the law of motion.

* The only friction in this economy is capital adjustment costs.

*Detailed Description*. This version differs from the version 2 model as follows:
1) The firm owns the capital stock and makes capital investment decisions. I allow for reversible investment (the firm can consume from its capital stock).
2) There are now two state variables (tfp and capital). 

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
Replicates Restuccia and Rogerson 2008 which uses an industry dynamics along the lines of Hopenhayn (1992) embedded with the neoclassical growth model. Firms are heterogenous in productivity. Productivity is constant over time and while entry is endogenous, there is only exogenous exit. (Note that my code Hopenhayn 1992 -- Version 2 is similar but has fluctuating productivity and endogenous exit).

*Detailed Description*. The authors show that resource misallocation across heterogenous firms can have sizeable negative effects on aggregate output and TFP even when policy does not relay on aggregate capital accumulation or aggregate relative price differences. The paper highlights the importance of resource misallocation across firms with different levels of productivity and could potentially explain cross-country differences in output per capita. The code calculates the efficient or benchmark economy and then compares  economies under a policy distortion of either output, capital or labor which reallocates resources among firms through tax/subsidies. Each firm faces its own tax or subdidy. To emphasize the effects, for each tax rate the code finds the subsidy rate that will generate the same aggregate capital stock as the benchmark economy. The focus is on policies that create idiosyncratic distortions to establishment-level decisions and hence cause a reallocation of resources across establishments. 
