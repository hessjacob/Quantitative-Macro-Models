# Quantitative-Macro-Models
This is a collection of code for quantitative macroeconomic models that I have written as personal learning exercises. All the codes, aside from the RANK block, have heterogenous agents and are written in python using numba. You can find different versions under each topic where I might use different computational methods, increase the scale of the model and/or adjust some key assumptions. I have worked hard to make them readable and very fast so that they might help others who are interested in learning about these topics. References used can be found in each file. Please remember to cite me if you use my codes.

**Quick Guide**
- Heterogenous households
  * Aiyagari
  * Consumption Saving (aka Income Flucuation Problem)
 
- Heterogenous firms/Industry dynamics
  * Hopenhayn
  * Restuccia and Rogerson
 
- Representative household
  * Neoclassical growth
  * RANK models

# Aiyagari 
Stationary equilibrium solution in a production economy with incomplete markets and no aggregate uncertainty. Heterogenous agents are infinitely lived and are exposed to idiosyncratic income risk. I solve the versions of the model with three different solution methods to solve the household problem. In addition, the user can choose from three different methods to calculate the stationary distribution. Version 2 can be used to replicate Aiyagari (1994). The labor supply is perfectly inelastic (exogenous).

**Solution Methods**

- Value Function Iteration with Discretization
    
- Policy Function Iteration on Euler Equation with Linear Interpolation

- Endogenous Grid Method
  
**Versions**
- Version 1 -- Two income states and a transition matrix both of which can be set by the user.
- Version 2 -- Continuous income process which is discretely approximated up to seven different income states using the Rouwenhorst method. The Tauchen method is available in the code should the user want an exact replication. 

**Code Features** 

1) The user can choose to find the stationary distribution with one of three methods:
   * Discrete approximation of the density function which conducts a fixed point iteration with linear interpolation
   * Eigenvector method to solve for the exact stationary density.
   * Monte carlo simulation with 50,000 households. 
2) Exogenous borrowing constraint which the user can choose. 
3) Calculation of the euler equation error both across the entire grid space and through a simulation.
4) Plots the capital supply and demand as functions of the capital stock. 


# Consumption Saving in Incomplete Markets (aka the income flucuation problem)
Partial equilibrium solution (prices are exogenously set) for heterogenous agents that are infinitely lived in incomplete markets and are exposed idiosyncratic income risk. Each numbered version under different solution methods solves the same problem. Version 1 solves a small consumption savings problem with two income states. Version 2 approximates the income process with the Rouwenhorst method and has more income states. These codes are extended to solve for general equilibrium in the Aiyagari section. 

**Solution Methods**

- Value Function Iteration with Discretization
    
- Policy Function Iteration on Euler Equation with Linear Interpolation

- Endogenous Grid Method
  
**Versions**
- Version 1 -- Two income states and a transition matrix both of which can be set by the user.
- Version 2 -- Continuous income process which is discretely approximated up to seven different income states using the Rouwenhorst method. 

**Code Features** 

1) The user can choose to find the stationary distribution with one of three methods:
   * Discrete approximation of the density function which conducts a fixed point iteration with linear interpolation
   * Eigenvector method to solve for the exact stationary density.
   * Monte carlo simulation with 50,000 households. 
2) Exogenous borrowing constraint which the user can choose. 
3) Calculation of the euler equation error both across the entire grid space and through a simulation.
  
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

* This is a frictionless economy and the equilibrium allocations are efficient.

*Detailed Description*.
The difference between this model and Hopenhayn (1992) and its general equilibrium extension in Hopenhayn and Rogerson (1993) is:

1) Unlike in either model, there is a flexible form of capital that the firm is able to rent from households (therefore they make the investment decision) as is standard in the neoclassical growth model. 
2) Labor is inelastically supplied by the household and not divisble like in Hopenhayn and Rogerson (1993).

Agents are infinitely lived and ex-ante identical. There are complete markets which allows me to to construct a repersentative household. Firms, on the other hand, are heterogenous in their productivity.The economy considered is similar to the neoclassical growth model except for output, which is produced by a large number of establishments subject to idiosyncratic productivity shocks that induce them to expand and contract over time.There is no aggregate uncertainty.Establishments have access to a decreasing returns to scale technology, pay a one-time fixed cost of entry, and a fixed cost of operation every period. 

There is ongoing exogenous and endogenous entry/exit in the steady state. Firms may exogenously die with probability lambda every period. I include this so that there are large productive firms that might suddenly shut down. The timing is as follows. At the beginning of every period firms that receive an exit shock leave the market. Remaining firms draw their new productivity and endogenously decide whether to continue or shut down. Those that continue choose the capital and labor factor demands to maximize profits. Meanwhile, there is a mass of potential entrants who draw an intial productivity level and decide whether they should enter the market. Before entering they must pay a one-time fixed entry cost.  

**Version 3 -- Firm Dynamics (Hopenhayn 1992) and the Neoclassical Growth Model (firm owns capital)**

* The code embeds the standard neoclassical growth model into Hopenhayn (1992) and solves for the stationary equilibrium in which there is continuous 
entry/exit of firms. The difference with this and version 2 is that the firm owns the capital stock and makes the investment decision. The model is a simplified general equilbrium extension of Clementi and Palazzo (2016). I solve the household problem using value function iteration and approximate the stationary joint distribution of firms by fixed point iteration on the law of motion.

* The only friction in this economy is capital adjustment costs. Absent of these frictions the equilibrium allocations are efficient.

*Detailed Description*. This version differs from the version 2 model as follows:
1) The firm owns the capital stock and makes capital investment decisions. I allow for reversible investment (the firm can consume from its capital stock).
2) There are now two state variables (tfp and capital). 

# Neoclassical Growth Model (Deterministic and Stochastic)
Social planner solution for the neoclassical growth model with inelastic labor supply (which coincides with the competitive equilibrium solution). Unless otherwise stated all solution methods are avaliable for both the deterministic and stochastic case. The versions differ in how the social planner problem is solved, but obtain the same solution. See the RBC files which is this model but with elastic labor supply.  

**Solution Methods**
- Value Function Iteration with Discretization
- Value Function Iteration with Linear Interpolation
  * Solved with conventional value function interation and interpolates with cubic splines to get the value for the value function between the grid points. The computation requires less iterations and is significantly faster than conventional value function iteration.

**Code Features** 

1) The deterministic files conduct a perfect foresight transition to the steady state.
2) The stochastic files generate a markov chain simulation and evaluate the accuracy of the solution by computing the euler equation errors.
3) The interpolation scheme (chosen by the user) for the transitions and simulations interpolates the decision rules with either cubic interpolation or by approximating them with chebyshev polynomials and OLS.


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


# Restuccia and Rogerson
Replicates Restuccia and Rogerson 2008 which uses an industry dynamics along the lines of Hopenhayn (1992) embedded with the neoclassical growth model. Firms are heterogenous in productivity. Productivity is constant over time and while entry is endogenous, there is only exogenous exit. (Note that my code Hopenhayn 1992 -- Version 2 is similar but has fluctuating productivity and endogenous exit).

*Detailed Description*. The authors show that resource misallocation across heterogenous firms can have sizeable negative effects on aggregate output and TFP even when policy does not relay on aggregate capital accumulation or aggregate relative price differences. The paper highlights the importance of resource misallocation across firms with different levels of productivity and could potentially explain cross-country differences in output per capita. The code calculates the efficient or benchmark economy and then compares  economies under a policy distortion of either output, capital or labor which reallocates resources among firms through tax/subsidies. Each firm faces its own tax or subdidy. To emphasize the effects, for each tax rate the code finds the subsidy rate that will generate the same aggregate capital stock as the benchmark economy. The focus is on policies that create idiosyncratic distortions to establishment-level decisions and hence cause a reallocation of resources across establishments. 
