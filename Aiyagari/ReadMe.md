
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
