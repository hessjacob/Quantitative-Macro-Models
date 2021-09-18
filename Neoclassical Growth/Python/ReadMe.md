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

