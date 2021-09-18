# Consumption Saving in Incomplete Markets (aka the income flucuation problem)
Partial equilibrium solution (prices are exogenously set) for heterogenous agents that are infinitely lived in incomplete markets and are exposed idiosyncratic income risk. Each numbered version under different solution methods solves the same problem. Version 1 solves a small consumption savings problem with two income states. Version 2 approximates the income process with the Rouwenhorst method and has more income states. These codes are extended to solve for general equilibrium in the Aiyagari section. 

**Solution Methods**

- Value Function Iteration with Discretization
    
- Policy Function Iteration on Euler Equation with Linear Interpolation

- Endogenous Grid Method
  
**Versions**
- Version 1 -- 2 income states. 
- Version 2 -- Continuous income process which is discretely approximated up to seven different income states using the Rouwenhorst method. 

**Code Features** 

1) The user can choose to find the stationary distribution with one of three methods:
   * Discrete approximation of the density function which conducts a fixed point iteration with linear interpolation
   * Eigenvector method to solve for the exact stationary density.
   * Monte carlo simulation with 50,000 households. 
2) Exogenous borrowing constraint which the user can choose. 
3) Calculation of the euler equation error both across the entire grid space and through a simulation.
