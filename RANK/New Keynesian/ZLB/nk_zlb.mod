var C           %consumption
    Pi          %gross inflation 
    N           %labor hours 
    A           %technology shock process
    Z           %Intertemporal preference shock 
    nu          %monetary policy shock process
    w           %real wage
    mc          %marginal cost
    vp          %price dispersion 
    Y           %output 
    Pi_star     %optimal reset price
    x1          %auxiliary variable 1 recursive price setting
    x2          %auxiliary variable 2 recursive price setting
    P           %price level 
    R           %nominal interest rate 
    r_real      %real interest rate
    yf          %flexible output
    log_Z log_A log_Y log_N log_w    %logged variables
    pi          %quarterly inflation rate
    i           %quarterly nominal interest rate
    ;

varexo eA eZ eNU;

parameters sigma alpha beta eta theta epsilon rho_a rho_z rho_R rho_nu psi phi_pi phi_y Rre

%steady state values
A_ss Z_ss vp_ss Pi_star_ss mc_ss R_ss P_ss Pi_ss r_real_ss N_ss C_ss w_ss Y_ss nu_ss x1_ss x2_ss yf_ss log_Z_ss log_A_ss log_Y_ss log_N_ss log_w_ss pi_ss i_ss

;

load param_nk_baseline;
set_param_value('sigma',sigma);
set_param_value('alpha',alpha);
set_param_value('beta',beta);
set_param_value('eta',eta);
set_param_value('theta',theta);
set_param_value('epsilon',epsilon);
set_param_value('rho_a',rho_a);
set_param_value('rho_z',rho_z);
set_param_value('rho_R',rho_R);
set_param_value('psi',psi);
set_param_value('rho_nu',rho_nu);
set_param_value('phi_pi',phi_pi);
set_param_value('phi_y',phi_y);
set_param_value('Rre',Rre);

%----------------------------------------------------------------
%  Steady State
%---------------------------------------------------------------

@#include "nk_ss.m"

initval;
C = C_ss;
Pi = Pi_ss;         
N = N_ss;          
A = A_ss;       
Z = Z_ss;
nu = nu_ss;        
w = w_ss;         
mc = mc_ss;       
vp = vp_ss;         
Y = Y_ss;           
Pi_star=Pi_star_ss;
x1=x1_ss;         
x2=x2_ss;         
P=P_ss;         
R=R_ss;           
r_real = r_real_ss;     
yf = yf_ss;
log_Z = log_Z_ss;
log_A = log_A_ss;   
log_Y = log_Y_ss;
log_N = log_N_ss;
log_w = log_w_ss;   
pi = pi_ss;
i = i_ss;              
  end;

%----------------------------------------------------------------
% Equilibrium Conditions
%----------------------------------------------------------------

model;
%(1) Euler Equation (eq 3 pg 54)
C^(-sigma) = beta*C(+1)^(-sigma)*R*(Pi(+1))^(-1)*(Z(+1)/Z);

%(2) Labor Supply (eq 2 pg 54)
psi*N^(eta) = C^(-sigma)*w;

%(3) Real Marginal Cost
mc = w/((1-alpha)*Y/N*vp); 

%(4) Resource Constraint
C = Y; 

%(5) Aggregate Production (above eq. 14 pg. 59)
Y=(A*N^(1-alpha))/vp;

%(6) Price Dispersion
vp = (1-theta)*Pi_star^(-epsilon/(1-alpha))+Pi^(epsilon/(1-alpha))*theta*vp(-1);
%vp = Pi^(epsilon)*((1-theta)*Pi_star^(-epsilon)+theta*vp(-1));

%(7) Law of Motion of Prices (eq 7 pg 55)
%1 = theta*Pi^(epsilon-1)+(1-theta)*Pi_star^(1-epsilon); jp
Pi^(1-epsilon) = (1-theta)*Pi_star^(1-epsilon)+theta;

%(8) Reset Price (unclear why JP has the exponent)
%Pi_star^(1+epsilon*(alpha/(1-alpha))) = epsilon/(epsilon-1)*(x1/x2); jp
Pi_star = Pi*epsilon/(epsilon-1)*(x1/x2);

%(9) Auxiliary Eq 1
x1 = Z*C^(-sigma)*mc*Y + theta*beta*Pi(+1)^(epsilon+alpha*epsilon/(1-alpha))*x1(+1);

%(10) Auxiliary Eq 2
x2 = Z*C^(-sigma)*Y + theta*beta*Pi(+1)^(epsilon-1)*x2(+1);

%(11) Price Level
Pi=P/P(-1);

%(12) Technology shock 
log(A) = rho_a*log(A(-1)) + eA;

%(13) Monetary Policy Shock
nu=rho_nu*nu(-1)+ eNU;

%(14) Discount Rate Shock 
log(Z) = rho_z*log(Z(-1))-eZ;

%(15) Fisher Equation
R=r_real*Pi(+1);

%(16) Flexible Output
yf = ((1/psi)*(epsilon-1)/epsilon)^(1/(sigma+eta))*A^((1+eta)/(sigma+eta));

%(17) Monetary policy
R=Rre;

% logged variables and other stuff
log_Z=log(Z);
log_A=log(A);
log_Y=log(Y);
log_N=log(N);
log_w=log(w);
pi = log(Pi);
i = log(R);

end;



