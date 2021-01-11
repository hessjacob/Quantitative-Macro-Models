%steady state values for mediumscale.mod

load param_mediumscale;

%Tech Shock
A_ss = 1;

%MEI Shock
Z_ss = 1;

%Inter Pref Shock
nu_ss = 1;

%gross inflation
Pi_ss = 1;

%price level
P_ss = 1;

%gross interest
R_ss = Pi_ss/beta;

%capital utilization (imposed)
u_ss = 1;

%multipliers are equal in SS. lambda_1_ss = lambda_2_ss from FOC
%investment. So from FOC Capital
rk_ss = 1/beta - (1-delta_0);

%rental rate capital. From FOC capital utilization
delta_1  = rk_ss - delta_2*(u_ss-1);

%reset price. From price law of motion
Pi_star_ss = ((Pi_ss^(1-epsilon_p) - theta_p*Pi_ss^(zeta_p*(1-epsilon_p)))/(1-theta_p))^(1/(1-epsilon_p));

%price dispersion. from price dispersion
vp_ss = ((1-theta_p)*(Pi_star_ss/Pi_ss)^(-epsilon_p))/(1-theta_p*Pi_ss^(epsilon_p*(1-zeta_p)));

%marginal cost. From x1/x2 ratio
mc_ss = ((epsilon_p-1)/epsilon_p)*(Pi_star_ss/Pi_ss)*((1-theta_p*beta*Pi_ss^(epsilon_p*(1-zeta_p)))/(1-theta_p*beta*Pi_ss^((epsilon_p-1)*(zeta_p-1))));

%from capital services equation. k=k_hat in ss since u=1

%capital labor ratio
k_nd_ratio = ((alpha*mc_ss)/rk_ss)^(1/(1-alpha));

%real wage. Comes from plugging in capital/labor ratio into marginal cost
%for rk.
w_real_ss = (1-alpha)*mc_ss*k_nd_ratio^(alpha);

%reset wage. from wage law of motion
w_star_ss = ((w_real_ss^(1-epsilon_w)*(1-theta_w*Pi_ss^((epsilon_w-1)*(1-zeta_w))))/(1-theta_w))^(1/(1-epsilon_w));


%wage disperion
vw_ss = ((1-theta_w)*(w_star_ss/w_real_ss)^(-epsilon_w*(1+eta)))/(1-theta_w*Pi_ss^(epsilon_w*(1+eta))*Pi_ss^(-zeta_w*epsilon_w*(1+eta)));

%consumption labor ratio. comes from profit function when F is picked such
%that profits are zero
c_nd_ratio = (1-omega)*(w_real_ss+rk_ss*k_nd_ratio) - delta_0*k_nd_ratio;

%steady state labor
psi_ss = 19.18;
Nd_ss = ((1/psi_ss)*((epsilon_w-1)/epsilon_w)*(1/c_nd_ratio)*((1-beta*b)/(1-b))*w_star_ss*((w_real_ss/w_star_ss)^(-epsilon_w*eta))* (((1-theta_w*beta*Pi_ss^(epsilon_w*(1+eta)*(1-zeta_w))))/(1-theta_w*beta*Pi_ss^((epsilon_w-1)*(1-zeta_w)))))^(1/(1+eta));

%fixed cost
F = Nd_ss*(k_nd_ratio^(alpha)-(w_real_ss+rk_ss*k_nd_ratio)*vp_ss);

%capital services 
K_hat_ss=k_nd_ratio*Nd_ss;
%K_hat_ss=((alpha*mc_ss)/rk_ss)^(1/(1-alpha))*Nd_ss; equivalently

%capital stock. from capital services equation
K_ss = K_hat_ss/u_ss;

%investment. from capital accumulation 
inv_ss = delta_0*K_ss;

%output from resource constraint 
Y_ss = (A_ss*K_hat_ss^(alpha)*Nd_ss^(1-alpha)-F)/vp_ss;
%Y_ss = (C_ss + inv_ss)/(1-omega);

%government 
G_ss = omega*Y_ss;

%consumption
C_ss = Y_ss - inv_ss - G_ss;
%C_ss = c_nd_ratio*Nd_ss; equivalently

%lambda1. from FOC consumption
lambda_1_ss = (1/C_ss)*((1-beta*b)/(1-b));

%lagrange multipliers. From FOC Investment
lambda_2_ss = lambda_1_ss;

%aggregate labor supply
N_ss = vw_ss*Nd_ss;

%output gap
Xg_ss = alpha*log(u_ss);

%price auxiliary eq 1
X1_ss = (lambda_1_ss*mc_ss*Y_ss)/(1-theta_p*beta*Pi_ss^(epsilon_p*(1-zeta_p)));

%price auxiliary eq 2
X2_ss = (lambda_1_ss*Y_ss)/(1-theta_p*beta*Pi_ss^((epsilon_p-1)*(zeta_p-1)));

%wage auxiliary eq 1
H1_ss = (psi_ss*(w_real_ss/w_star_ss)^(epsilon_w*(1+eta))*Nd_ss^(1+eta))/(1-theta_w*beta*Pi_ss^(epsilon_w*(1+eta)*(1-zeta_w)));

%wage auxiliary eq 2
H2_ss = (lambda_1_ss*(w_real_ss/w_star_ss)^(epsilon_w)*Nd_ss)/(1-theta_w*beta*Pi_ss^((epsilon_w-1)*(1-zeta_w)));

%ngdp
NY_ss = P_ss*Y_ss;

%welfare
Welfare_ss = ((log(C_ss - b*C_ss) - psi_ss*vw_ss*(Nd_ss^(1+eta)/(1+eta))))/(1-beta);

%logged variables
log_A_ss = log(A_ss);
log_Z_ss = log(Z_ss);
log_G_ss = log(G_ss);
log_psi_ss = log(psi_ss);

log_nu_ss = log(nu_ss);
pi_ss = log(Pi_ss);
i_ss = log(R_ss);
log_Y_ss = log(Y_ss);
log_C_ss = log(C_ss);
log_Nd_ss = log(Nd_ss);
log_I_ss = log(inv_ss);
log_w_ss = log(w_real_ss);


dY_ss = 0;
dC_ss = 0;
dNd_ss = 0;
dI_ss = 0;
dw_ss = 0;



