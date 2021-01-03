# Quantitative-Macro-Models
This is a collection of code for quantitative macroeconomic models that I have written or modified from someone else. The models in the same folder but written in different coding languages will do the same thing. 

You are welcome to download and use anything here.

# File Content

## Consumption Saving Problem (aka the income flucuation problem)
Partial equilibrium solution for the infinitely lived household. 

- Value Function Iteration
  * Version 1 -- 2 income states. Solves the household problem with VFI. Runs a simulation for 50,000 households.  
  * Version 2 -- Continuous income process which is discretely approximated up to seven different income states using the Rouwenhorst method. Solves the household    problem with VFI. Runs a simulation for 50,000 households
  
- Endogenous Grid Method
  * Version 1 -- 2 income states. Solves the household problem with EGM. Runs a simulation for 50,000 households. Solves for the invariant distribution using an eigenvector method. 

## Neoclassical Growth (Deterministic and Stochastic)
- Discretized VFI to solve the model and Chebyshev polynomial approximation of decision rules to simulate.
- Computes the solution and does a simulation.
