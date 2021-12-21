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
