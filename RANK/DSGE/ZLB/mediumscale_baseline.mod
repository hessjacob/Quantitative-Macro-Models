%Calibrated medium-scale DSGE based on Sim's NGDP, Sim's notes and JFV 2006. No ZLB. Capital utilization is a little different here compared to notes. 


var C               %consumption  
    nu              %intertemporal shock process  
    lambda_1        %lagrange multiplier 1 on hh flow budget constraint
    lambda_2        %lagrange multiplier 2 on capital accumulation
    rk              %rental rate of capital
    Pi              %gross inflation
    R               %nominal gross interest
    u               %capacity utilization
    inv             %investment
    Z               %marginal efficiency of investment (MEI) shock process
    w_star          %reset wage
    H1              %Auxiliary wage eq 1
    H2              %Auxiliary wage eq 2
    psi             %intratemporal shock process
    Nd              %homogeneous labor input available for production offered by union
    N               %aggregate labor supply
    w_real          %real wage
    K_hat           %capital services
    K               %capital stock
    mc              %real marginal cost
    A               %technology shock process
    X1              %price auxilliary eq 1
    X2              %price auxilliary eq 2 
    Pi_star         %price reset
    Y               %aggregate output
    G               %government consumption
    vp              %price dispersion
    vw              %wage dispersion
    Xg              %estimated output gap
    P               %price level
    NY              %nominal gdp
    Welfare         %welfare

%logged variables
log_A log_Z log_G log_psi log_nu 
dY dC dNd dI dw 
log_Y log_C log_Nd log_I log_w
    ;

varexo eA eZ eNU eG ePSI eM;

parameters     beta                 %discount factor
               b                    %habit persistence parameter 
               delta_0              %depreciation rate
               delta_1              %capital utilization, linear term
               delta_2              %capital utilization, quadratic term
               kappa                %capital adjustment cost parameter
               epsilon_w            %labor elasticity of substitution
               theta_w              %wage calvo parameter
               eta                  %inverse frisch elasticity
               zeta_w               %wage indexation
               zeta_p               %price indexation
               alpha                %capital share         
               theta_p              %calvo price parameter
               epsilon_p            %elasticity of substitution goods
               rho_g                %autocorrelation government consumption 
               rho_r                %autocorrelation interest 
               rho_a                %autocorrelation technology shock
               rho_z                %autocorrelation MEI
               rho_nu               %autocorrelation intertemporal preference
               rho_psi              %autocorrelation intratemporal preference
               omega                %steady state government consumption
             
               phi_pi               %inflation feedback
               phi_y                %output growth feedback
               phi_x                %output gap feedback
               Rre                  %gross interest rate lower bound

%steady state values
C_ss nu_ss lambda_1_ss lambda_2_ss rk_ss Pi_ss R_ss u_ss inv_ss Z_ss w_star_ss H1_ss H2_ss psi_ss Nd_ss N_ss w_real_ss K_hat_ss K_ss mc_ss A_ss
X1_ss X2_ss Pi_star_ss Y_ss G_ss F vp_ss vw_ss Xg_ss P_ss NY_ss log_A_ss log_Z_ss log_G_ss log_psi_ss log_nu_ss dY_ss dC_ss 
dNd_ss dI_ss dw_ss Welfare_ss log_Y_ss log_C_ss log_I_ss log_w_ss log_Nd_ss       

               ;

load param_mediumscale;

set_param_value('beta',beta);
set_param_value('b',b);
set_param_value('delta_0',delta_0);
set_param_value('delta_2',delta_2);
set_param_value('kappa',kappa);
set_param_value('epsilon_w',epsilon_w);
set_param_value('theta_w',theta_w);
set_param_value('eta',eta);
set_param_value('zeta_w',zeta_w);
set_param_value('zeta_p',zeta_p);
set_param_value('alpha',alpha);
set_param_value('theta_p',theta_p);
set_param_value('epsilon_p',epsilon_p);
set_param_value('rho_g',rho_g);
set_param_value('rho_r',rho_r);
set_param_value('rho_a',rho_a);
set_param_value('rho_z',rho_z);
set_param_value('rho_nu',rho_nu);
set_param_value('rho_psi',rho_psi);
set_param_value('omega',omega);
set_param_value('phi_pi',phi_pi);
set_param_value('phi_y',phi_y);
set_param_value('phi_x',phi_x);
set_param_value('Rre',Rre);

%----------------------------------------------------------------
%  Steady State
%---------------------------------------------------------------

@#include "mediumscale_ss_zlb.m"

initval;
    C=C_ss;             
    nu=nu_ss;            
    lambda_1 = lambda_1_ss;        
    lambda_2 = lambda_2_ss;       
    rk=rk_ss;              
    Pi=Pi_ss;             
    R=R_ss;              
    u=u_ss;              
    inv=inv_ss;           
    Z=Z_ss;              
    w_star=w_star_ss;          
    H1=H1_ss;              
    H2=H2_ss;             
    psi=psi_ss;            
    Nd=Nd_ss;             
    N=N_ss;               
    w_real=w_real_ss;       
    K_hat = K_hat_ss;          
    K = K_ss;               
    mc = mc_ss;            
    A = A_ss;               
    X1 = X1_ss;              
    X2 = X2_ss;             
    Pi_star = Pi_star_ss;       
    Y = Y_ss;            
    G = G_ss;            
    vp = vp_ss;             
    vw = vw_ss;             
    Xg = Xg_ss;             
    P  = P_ss;     
    NY = NY_ss;
    Welfare = Welfare_ss;         
    log_A = log_A_ss; 
    log_Z = log_Z_ss;
    log_G = log_G_ss;
    log_psi = log_psi_ss;
    log_nu = log_nu_ss;
    log_Y = log_Y_ss;
    log_C = log_C_ss;
    log_I = log_I_ss;
    log_w = log_w_ss;
    log_Nd = log_Nd_ss;
    dY = dY_ss;
    dC = dC_ss;
    dNd = dNd_ss;
    dI = dI_ss;
    dw = dw_ss;
end;

                
%----------------------------------------------------------------
% Equilibrium Conditions
%----------------------------------------------------------------

model;
%(1) FOC consumption
lambda_1 = nu/(C-b*C(-1)) - beta*b*nu(+1)/(C(+1)-b*C);

%(2) FOC capital utilization
rk=delta_1+delta_2*(u-1);

%(3) FOC bonds
lambda_1=beta*lambda_1(+1)*R*Pi(+1)^(-1);

%(4) FOC capital 
%note: JFV: Big Q is the multiplier (my lambda_2). When divided by lambda_1, Q/lambda = q the ratio is Tobin's q (the value of installed capital in terms of its replacement cost). 
%My equation does not do this. This is from pg. 4 baseline JFV and Sim's NGDP

lambda_2=beta*(lambda_1(+1)*(rk(+1)*u(+1)-(delta_1*(u(+1)-1)+delta_2/2*(u(+1)-1)^2))+lambda_2(+1)*(1-delta_0)); 

%(5) FOC investment
lambda_1=lambda_2*Z*(1-(kappa/2*(inv/inv(-1)-1)^2)-(kappa*(inv/inv(-1)-1)*(inv/inv(-1))))+beta*lambda_2(+1)*Z(+1)*kappa*(inv(+1)/inv-1)*(inv(+1)/inv)^2;

%(6) Wage Reset
w_star = epsilon_w/(epsilon_w-1) * (H1/H2);

%(7) Auxiliary Wage Eq 1
H1 = psi*nu*(w_star/w_real)^(-epsilon_w*(1+eta))*Nd^(1+eta) + theta_w*beta*(w_star(+1)/w_star)^(epsilon_w*(1+eta))*Pi^(-zeta_w*epsilon_w*(1+eta))*Pi(+1)^(epsilon_w*(1+eta))*H1(+1);

%(8) Auxiliary Wage Eq 2
H2=lambda_1*(w_star/w_real)^(-epsilon_w)*Nd+theta_w*beta*(w_star(+1)/w_star)^(epsilon_w)*Pi^(zeta_w*(1-epsilon_w))*Pi(+1)^(epsilon_w-1)*H2(+1);

%(9) Capital-Labor Ratio (optimal input ratio)
K_hat/Nd = (alpha/(1-alpha))*(w_real/rk);

%(10) Capital Services
K_hat=u*K(-1);

%(11) Real Marginal Cost
mc = (w_real^(1-alpha)*rk^(alpha))/A*(1-alpha)^(alpha-1)*alpha^(-alpha);

%(12) Price Reset
Pi_star=epsilon_p/(epsilon_p-1)*Pi*(X1/X2);

%(13) Price Auxiliary Eq 1
X1 = lambda_1*mc*Y+theta_p*beta*Pi^(-zeta_p*epsilon_p)*Pi(+1)^(epsilon_p)*X1(+1);

%(14) Price Auxiliary Eq 2
X2 = lambda_1*Y + theta_p*beta*Pi^(zeta_p*(1-epsilon_p))*Pi(+1)^(epsilon_p-1)*X2(+1);

%(15) Resource Constraint
Y = C + inv + G + (delta_1*(u-1)+(delta_2/2)*(u-1)^2)*K(-1);

%(16) Capital Law of Motion
K = Z*(1-(kappa/2)*((inv/inv(-1))-1)^2)*inv+(1-delta_0)*K(-1); 

%(17) Aggregate Production Function 
Y = (A*K_hat^(alpha)*Nd^(1-alpha)-F)/vp;

%(18) Price Law of Motion
Pi^(1-epsilon_p) = (1-theta_p)*Pi_star^(1-epsilon_p) + theta_p*Pi(-1)^(zeta_p*(1-epsilon_p));

%(19) Wage Law of Motion
w_real^(1-epsilon_w) = (1-theta_w)*w_star^(1-epsilon_w)+theta_w*Pi(-1)^(zeta_w*(1-epsilon_w))*Pi^(epsilon_w-1)*w_real(-1)^(1-epsilon_w);

%(20) Price Dispersion
vp = Pi^(epsilon_p)*((1-theta_p)*Pi_star^(-epsilon_p) + theta_p*Pi(-1)^(-epsilon_p*zeta_p)*vp(-1));

%(21) Wage Dispersion
vw = (1-theta_w)*(w_star/w_real)^(-epsilon_w*(1+eta))+theta_w*(w_real/w_real(-1)*Pi)^(epsilon_w*(1+eta))*Pi(-1)^(zeta_w*epsilon_w*(1+eta))*vw(-1);

%(22) Government 
log(G) = (1-rho_g)*log(G_ss) + rho_g*log(G(-1)) + eG;

%(23) Estimated Output Gap (Gust et al. 2019)
Xg = alpha*log(u) + (1-alpha)*log(N/N_ss);

%(24) Monetary Policy
%log(R/R_ss) = rho_r*log(R(-1)/R_ss) + (1-rho_r)*(phi_pi*log(Pi/Pi_ss)+phi_x*Xg+phi_y*log(Y/Y(-1)))+eM;
log(R) = (1-rho_r)*log(R_ss) + rho_r*log(R(-1)) + (1-rho_r)*(phi_pi*log(Pi/Pi_ss)+phi_x*Xg+phi_y*log(Y/Y(-1)))+eM;

%(25) Technology AR(1) Process
log(A)=rho_a*log(A(-1))+eA;

%(26) MEI AR(1) Process
log(Z)=rho_z*log(Z(-1))+eZ;

%(27) Intertemporal Pref AR(1) Process
log(nu) = rho_nu*log(nu(-1))+eNU;

%(28) Intratemporal Pref AR(1) Process
log(psi) = (1-rho_psi)*log(psi_ss)+rho_psi*log(psi(-1))+ePSI;

%(29) Aggregate Labor Supply
N = vw*Nd;

%(30) Welfare
Welfare = nu*(log(C - b*C(-1)) - psi*vw*(Nd^(1+eta)/(1+eta)))+beta*Welfare(+1);

%(31) Price Level
Pi = P/P(-1);

%(32) NGDP
NY = P*Y;


%logged variables and other stuff
log_A = log(A);
log_Z = log(Z);
log_G = log(G);
log_psi = log(psi);
log_nu = log(nu);

log_Y = log(Y);
log_C = log(C);
log_I = log(inv);
log_w = log(w_real);
log_Nd = log(Nd);


dY = log(Y) - log(Y(-1));
dC = log(C) - log(C(-1));
dNd = log(Nd) - log(Nd(-1));
dI = log(inv) - log(inv(-1));
dw = log(w_real) - log(w_real(-1));

end;

%steady state check
resid(1);
steady;
check;

%----------------------------------------------------------------
%  define shock variances
%---------------------------------------------------------------

%standard deviations from sims 2016 notes.


shocks;

var eA;     stderr 0.01;    

var eNU;    stderr 0.01;
                    
var eZ;     stderr 0.025;

var eG;      stderr 0.01;

var ePSI;    stderr 0.005;

var eM;      stderr 0.003;

end;

stoch_simul(order=1,irf=20,nograph);

               
              




