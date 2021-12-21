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
entry/exit of firms. The difference with this and version 2 is that the firm owns the capital stock and makes the investment decision. The model is a simplified general equilbrium extension of Clementi and Palazzo (2016). I solve the household problem using value function iteration and approximate the stationary joint distribution of firms by fixed point iteration on the law of motion.

* The only friction in this economy is capital adjustment costs.

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
