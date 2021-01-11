close all
clear

%calibration from Sims NGDP priors.
               beta=0.995;                 %discount factor
               b=0.7;                    %habit persistence parameter 
               delta_0 = 0.025;              %depreciation rate
               delta_2 = 0.01;             %capital utilization, quadratic term
               kappa=4;                %capital adjustment cost parameter
               epsilon_w=11;            %labor elasticity of substitution
               theta_w=0.75;              %wage calvo parameter
               eta=2;                  %inverse frisch elasticity
               zeta_w=0;               %wage indexation
               zeta_p=0;              %price indexation
               alpha=1/3;                %capital share         
               theta_p=0.75;              %calvo price parameter
               epsilon_p=11;            %elasticity of substitution goods
               rho_g=0.95;                %autocorrelation government consumption 
               rho_r=0.8;                %autocorrelation interest 
               rho_a=0.95;                %autocorrelation technology shock
               rho_z=0.8;                %autocorrelation MEI
               rho_nu=0.8;               %autocorrelation intertemporal preference
               rho_psi=0.8;              %autocorrelation intratemporal preference
               omega=0.2;                %steady state government consumption
             
               phi_pi=1.5;               %inflation feedback
               phi_y=0.25;                %output growth feedback
               phi_x=0.05;                %output gap feedback
               Rre=1;                    %gross interest rate lower bound
               
save param_mediumscale  beta b delta_0 delta_2 kappa epsilon_w theta_w eta zeta_w zeta_p alpha theta_p epsilon_p rho_g rho_r rho_a rho_z rho_nu rho_psi omega phi_pi phi_y phi_x Rre              

mediumscale_ss_zlb

global M_ oo_

addpath('/Applications/Dynare/4.5.7/matlab/occbin_20140630/toolkit_files')
nperiods=50;
maxiter=50;
tol0 = 1e-8;

modnam = 'mediumscale_baseline';
modnamstar = 'mediumscale_zlb nostrict';    %add nostrict here because eM is part of the the system but not in any equation in the zlb mod. This skips over the warning.

constraint = 'R<Rre-Pi_ss/beta';
constraint_relax = 'R>Rre-Pi_ss/beta';

% Pick innovation for IRFs
irfshock =char('eA','eZ','eNU','eG','ePSI','eM');






% Solve the nonlinear model

%var eA;     stderr 0.01;    
%var eNU;    stderr 0.01;                   
%var eZ;     stderr 0.025;
%var eG;      stderr 0.01;
%var ePSI;    stderr 0.005;
%var eM;      stderr 0.003;


SHOCKS = [ zeros(2,6)
   0.06 0 0 0 0 0
  zeros(17,6) ] ;

shockssequence = SHOCKS;


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

%%

figure(1)
subplot(3,2,1)
plot(100*Y_p/Y_ss,'k','Linewidth',1.5)
title('Output')
ylabel('%  dev.from s.s.')
%hold on
%plot(100*yf_eA/yf_ss,'--k','Linewidth',1.5)
%hold off
grid on

subplot(3,2,2)
plot(400*(Pi_p+Pi_ss-1),'k','Linewidth',1.5)
title('Inflation (annualized)')
ylabel('Annualized Level, PPt')
%plot(400*Pi_eNU/Pi_ss,'k','Linewidth',1.5)
%title('Inflation (annualized)')
%ylabel('% dev.from s.s.')
grid on


subplot(3,2,3)
plot(4*100*(R_p+R_ss-1),'k','Linewidth',1.5)
title('Interest Rate (annualized)')
ylabel('Level in % pt.')
grid on

subplot(3,2,4)
plot(100*(C_p/C_ss),'k','Linewidth',1.5)
title('Consumption')
ylabel('%  dev.from s.s.')
grid on

subplot(3,2,5)
plot(Xg_p,'k','Linewidth',1.5)
title('Output Gap')
ylabel('abs. dev.from s.s.')
grid on

subplot(3,2,6)
plot(log_A_p,'k','Linewidth',1.5)
title('Technology Shock')
ylabel('Level')
grid on

