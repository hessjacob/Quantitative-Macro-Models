close all
clear


%set parameters

sigma = 1;
alpha = 0;
beta = 0.995;
eta = 1;
theta = 9/10;   %see description
epsilon = 6;
rho_a = 0.9;
rho_z = 0.8;
rho_R = 0;      %difficulty reaching zlb when not set to zero
rho_nu = 0.8;
psi = 1; 

%monetary policy 
phi_pi = 2.5;   %see description
phi_y=0.25;
Rre=1;

%last paragraph of calibration section pg. 35
%"To curb the volatility of inflation with a nod to empirical realism, we push this parameter from 0.75 to 0.9. This change, in
%conjunction with an interest rate rule that responds aggressively to inflation, with a coefficient ?? set at 2.5, prevents large
%disinflations from occurring at the zero lower bound, in line with recent
%U.S. experience."

save param_nk_baseline sigma alpha beta eta theta epsilon rho_a rho_z rho_R rho_nu phi_pi phi_y Rre psi;

nk_ss

global M_ oo_

addpath('/Applications/Dynare/4.5.7/matlab/occbin_20140630/toolkit_files')
nperiods=100;
maxiter=50;
tol0 = 1e-8;

modnam = 'nk_baseline';
modnamstar = 'nk_zlb';

constraint = 'R<Rre-Pi_ss/beta';
constraint_relax = 'R>Rre-Pi_ss/beta';

% Pick innovation for IRFs
irfshock =char('eA','eZ','eNU');






% Solve the nonlinear model

%original calibriation 
%10 std dev tech=0.1
%20 std dev intertemp = (20)*0.005 = 0.1
%4 std dev monetary = (4)*0.0025 = -0.01

%irfs
SHOCKS = [ zeros(2,3)
   0.1 0 0
  zeros(17,3) ] ;
shockssequence = SHOCKS;

%to run a stochastic simulation
%randn('seed',3);
%shockssequence = [1*randn(nperiods,1)*0.01 1*randn(nperiods,1)*0.005 1*randn(nperiods,1)*0.0025];



% Solve model, generate model IRFs
[zdatalinear zdatapiecewise zdatass oobase_ Mbase_  ] = ...
  solve_one_constraint(modnam,modnamstar,...
  constraint, constraint_relax,...
  shockssequence,irfshock,nperiods);

% unpack the IRFs  
for i=1:M_.endo_nbr
  eval([deblank(M_.endo_names(i,:)),'_u=zdatalinear(:,i);']);
  eval([deblank(M_.endo_names(i,:)),'_p=zdatapiecewise(:,i);']);
  eval([deblank(M_.endo_names(i,:)),'_s=zdatass(i);']);
end

%% Generate irfs

t=1:1:nperiods;

figure(1)

subplot(3,2,1)
plot(t,4*100*(Pi_p+Pi_ss-1),'k',t,4*100*(Pi_u+Pi_ss-1),':k','Linewidth',1.5)
%'% pt. Level'
%plot(t,pi_ann_p,'k',t,pi_ann_u,'--k','Linewidth',1.5) equivalent
title('Inflation (annualized)')
ylabel('Level in % pt.')
%ylabel('% dev.from s.s.')
grid on
legend('piecewise','linear','Location','southeast')

subplot(3,2,2)
plot(t,100*Y_p/Y_s,'k',t,100*Y_u/Y_s,':k','Linewidth',1.5)
title('Output')
ylabel('%  dev.from s.s.')
grid on

subplot(3,2,3)
plot(t,400*(R_p+R_s-1),'k',t,400*(R_u+R_s-1),':k','Linewidth',1.5)
title('Interest Rate (annualized)')
ylabel('Level in % pt.')
grid on

subplot(3,2,4)
plot(t,100*(N_p/N_s),'k',t,100*(N_u/N_s),':k','Linewidth',1.5)
title('Labor Hours')
ylabel('%  dev.from s.s.')
grid on

subplot(3,2,5)
plot(t,100*(w_p/w_s),'k',t,100*(w_u/w_s),':k','Linewidth',1.5)
title('Real Wage')
ylabel('%  dev.from s.s.')
grid on

subplot(3,2,6)
plot(t,log_A_p,'k','Linewidth',1.5)
title('Shock')
ylabel('Level')
grid on

saveas(gcf,'tech_shock_zlb','epsc')

