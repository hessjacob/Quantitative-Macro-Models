New Keynesian -- The standard model with calvo price rigidities.

DSGE -- Medium Scale DSGE model based in Christiano et. al (2005) which closely follows Eric Sims' notes (which I have included). The is calibrated to match (pre-2008 crisis) data, which is also cleaned and included. Using dynare you can easily take this model and my cleaned data and use Bayesian estimation techniques. 

Both models are extended and have an occasionally binding constraint on the nominal interest rate. 


These models are solved using 

- Dynare (download at https://www.dynare.org/download/) which is run through Matlab. I used version 4.5.7 and MATLAB_2016B 

- In the ZLB folders in addition to Dynare, Occbin toolbox version occbin_20140630 (download at https://www.matteoiacoviello.com/research.htm) which is a collection of matlab files that help solve linearized models with an occassionally binding constraint(s) on endogenous variable(s).



 