load param_nk;


%(1) tech 
A_ss = 1;

%(2) preference
Z_ss = 1; 

%(3) gross inflation
Pi_ss = 1;

%(4) reset price
Pi_star_ss = ((Pi_ss^(1-epsilon)-theta)/(1-theta))^(1/(1-epsilon)); 

%(5) marginal cost
mc_ss = (epsilon-1)/epsilon * ((1-theta*beta*Pi_ss^(epsilon))/(1-theta*beta*Pi_ss^(epsilon-1))) * (Pi_star_ss/Pi_ss);

%(6) price dispersion 
vp_ss = (1-theta)*(Pi_ss/Pi_star_ss)^(epsilon)*(1/(1-Pi_ss^(epsilon)*theta)); 

%(7) price level
P_ss = Pi_ss;

%(8) gross interest 
R_ss = Pi_ss/beta;

%(9) real interest 
r_real_ss = R_ss/Pi_ss;

%(10) labor hours
N_ss = ((1/psi)*(1-alpha)*mc_ss*vp_ss^(sigma))^(1/((1-sigma)*alpha+eta+sigma)); 
%N_ss = (vp_ss^(sigma)*mc_ss)^(1/(eta+sigma)); with linear prod function

%(11) consumption
C_ss = (A_ss*N_ss^(1-alpha))/vp_ss;

%(12) real wage
w_ss = C_ss^(sigma)*N_ss^(eta);

%(13) resource constraint
Y_ss = C_ss;

%(14) monetary shock
nu_ss = 0;

%(15) auxiliary eq 1
x1_ss = C_ss^(-sigma)*Y_ss*mc_ss/(1-beta*theta*Pi_ss^(epsilon/(1-alpha)));

%(16) auxiliary eq 2 
x2_ss = C_ss^(-sigma)*Y_ss/(1-beta*theta*Pi_ss^(epsilon-1));

%(17) flexible output
yf_ss = ((1/psi)*(epsilon-1)/epsilon)^(1/(sigma+eta));

%logged variables (2 in total)
log_Z_ss = 0;
log_A_ss = 0;
log_Y_ss = log(Y_ss);
log_N_ss = log(N_ss);
log_w_ss = log(w_ss);
pi_ss = log(Pi_ss);
i_ss = log(R_ss);


