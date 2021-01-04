# Quantitative-Macro-Models
This is a collection of code for quantitative macroeconomic models that I have written or modified from someone else. The models in the same folder but written in different coding languages will do the same thing. 

You are welcome to download and use anything here.

# File Content

## Consumption Saving in Incomplete Markets (aka the income flucuation problem)
Partial equilibrium solution (prices are exogenously set) for the infinitely lived household with incomplete markets and idiosyncratic income risk. The versions differ in how the household problem is solved and how the income shock process is specified. These codes are extended to solve in general equilibrium in the Aiyagari section. 

Shared features in all versions is 

1) Runs a markov chain simulation for 50,000 households and the stationary distribution is approximated. 
2) Exogenous borrowing constraint which the user can choose. 

Household Solution Methods

- Value Function Iteration
  * Version 1 -- 2 income states. 
  * Version 2 -- Continuous income process which is discretely approximated up to seven different income states using the Rouwenhorst method. 
  
- Endogenous Grid Method
  * Version 1 -- 2 income states. 

## Neoclassical Growth (Deterministic and Stochastic)
- Discretized VFI to solve the model and Chebyshev polynomial approximation of decision rules to simulate.
- Computes the solution and does a simulation.
