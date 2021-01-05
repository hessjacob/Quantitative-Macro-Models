# Quantitative-Macro-Models
This is a collection of code for quantitative macroeconomic models that I have written as personal learning exercises. Mostly all the codes involve heterogenous agents and are written in python using numba. References used can be found in each file.    

You are welcome to download and use anything here!

# File Content

## Aiyagari 
General equilibrium in a production economy with incomplete markets. Heterogenous agents are infinitely lived and are exposed to idiosyncratic income risk. The versions differ in how the household problem is solved, how the income shock process is specified and how the stationary distribution is approximated.

**Shared features in all versions is** 

1) Plots the wealth distribution, capital supply and demand and policy functions.
2) Exogenous borrowing constraint which the user can choose. 
3) Unless otherwise specified, a monte carlo simulation is used to approximate the stationary distribution

**Codes and Solution Methods**

- Value Function Iteration
  * Version 1 -- 2 income states. 
  * Version 2 -- Code will replicate Aiyagari (1994). A slight difference is that the continuous income process which is discretely approximated up to seven different income states using the Rouwenhorst method rather than Tauchen. The Tauchen method is available in the code should the user want an exact replication. 
  

## Consumption Saving in Incomplete Markets (aka the income flucuation problem)
Partial equilibrium solution (prices are exogenously set) for the infinitely lived household with incomplete markets and idiosyncratic income risk. The versions differ in how the household problem is solved and how the income shock process is specified. These codes are extended to solve for general equilibrium in the Aiyagari section. 

**Shared features in all versions is** 

1) Runs a markov chain simulation for 50,000 heterogenous households and the stationary distribution is approximated. 
2) Exogenous borrowing constraint which the user can choose. 

**Codes and Solution Methods**

- Value Function Iteration
  * Version 1 -- 2 income states. 
  * Version 2 -- Continuous income process which is discretely approximated up to seven different income states using the Rouwenhorst method. 
  
- Endogenous Grid Method
  * Version 1 -- 2 income states. 

## Neoclassical Growth (Deterministic and Stochastic)
- Discretized VFI to solve the model and Chebyshev polynomial approximation of decision rules to simulate.
- Computes the solution and does a simulation.
