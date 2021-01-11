close all
clear 


%set parameters

sigma = 1;
alpha = 0;
beta = 0.995;
eta = 1;
theta = 3/4;
epsilon = 10;
rho_a = 0.9;
rho_z = 0.8;
rho_R = 0.8;
rho_nu = 0.8;
psi = 1;

%monetary policy 
phi_pi = 1.5;
phi_y=0.125;


save param_nk sigma alpha beta eta theta epsilon rho_a rho_z rho_R rho_nu psi phi_pi phi_y;

dynare new_keynesian noclearall nolog

% generate irfs 
%shock variable difference between shocked variable and ss



figure(1)
subplot(3,2,1)
plot(100*Y_eA/Y_ss,'k','Linewidth',1.5)
title('Output')
ylabel('%  dev.from s.s.')
%hold on
%plot(100*yf_eA/yf_ss,'--k','Linewidth',1.5)
%hold off
grid on

subplot(3,2,2)
plot(400*(Pi_eA+Pi_ss-1),'k','Linewidth',1.5)
title('Inflation (annualized)')
ylabel('Annualized Level, PPt')
%plot(400*Pi_eNU/Pi_ss,'k','Linewidth',1.5)
%title('Inflation (annualized)')
%ylabel('% dev.from s.s.')
grid on


subplot(3,2,3)
plot(4*100*(R_eA+R_ss-1),'k','Linewidth',1.5)
title('Interest Rate (annualized)')
ylabel('Level in % pt.')
grid on

subplot(3,2,4)
plot(100*(N_eA/N_ss),'k','Linewidth',1.5)
title('Labor Hours')
ylabel('%  dev.from s.s.')
grid on

subplot(3,2,5)
plot(100*((Y_eA/Y_ss)-(yf_eA/yf_ss)),'k','Linewidth',1.5)
title('Output Gap')
ylabel('%  dev.from s.s.')
grid on

subplot(3,2,6)
plot(log_A_eA,'k','Linewidth',1.5)
title('Technology Shock')
ylabel('Level')
grid on

saveas(gcf,'tech_shock_nk','epsc')

% intertemporal preference shock irf

figure(2)
subplot(3,2,1)
plot(100*Y_eZ/Y_ss,'k','Linewidth',1.5)
title('Output')
ylabel('%  dev.from s.s.')
%hold on
%plot(100*yf_eA/yf_ss,'--k','Linewidth',1.5)
%hold off
grid on

subplot(3,2,2)
plot(400*(Pi_eZ+Pi_ss-1),'k','Linewidth',1.5)
title('Inflation (annualized)')
ylabel('Annualized Level, PPt')
%plot(400*Pi_eNU/Pi_ss,'k','Linewidth',1.5)
%title('Inflation (annualized)')
%ylabel('% dev.from s.s.')
grid on


subplot(3,2,3)
plot(4*100*(R_eZ+R_ss-1),'k','Linewidth',1.5)
title('Interest Rate (annualized)')
ylabel('Level in % pt.')
grid on

subplot(3,2,4)
plot(100*(N_eZ/N_ss),'k','Linewidth',1.5)
title('Labor Hours')
ylabel('%  dev.from s.s.')
grid on

subplot(3,2,5)
plot(100*((Y_eZ/Y_ss)-(yf_eZ/yf_ss)),'k','Linewidth',1.5)
title('Output Gap')
ylabel('%  dev.from s.s.')
grid on

subplot(3,2,6)
plot(log_Z_eZ,'k','Linewidth',1.5)
title('Preference Shock')
ylabel('Level')
grid on

saveas(gcf,'pref_shock_nk','epsc')

% policy shock irf

figure(3)
subplot(3,2,1)
plot(100*Y_eNU/Y_ss,'k','Linewidth',1.5)
title('Output')
ylabel('%  dev.from s.s.')
%hold on
%plot(100*yf_eA/yf_ss,'--k','Linewidth',1.5)
%hold off
grid on

subplot(3,2,2)
plot(400*(Pi_eNU+Pi_ss-1),'k','Linewidth',1.5)
title('Inflation (annualized)')
ylabel('Annualized Level, PPt')
%plot(400*Pi_eNU/Pi_ss,'k','Linewidth',1.5)
%title('Inflation (annualized)')
%ylabel('% dev.from s.s.')
grid on


subplot(3,2,3)
plot(4*100*(R_eNU+R_ss-1),'k','Linewidth',1.5)
title('Interest Rate (annualized)')
ylabel('Annualized Level in % pt.')
grid on

subplot(3,2,4)
plot(100*(N_eNU/N_ss),'k','Linewidth',1.5)
title('Labor Hours')
ylabel('%  dev.from s.s.')
grid on

subplot(3,2,5)
plot(100*((Y_eNU/Y_ss)-(yf_eNU/yf_ss)),'k','Linewidth',1.5)
title('Output Gap')
ylabel('%  dev.from s.s.')
grid on

subplot(3,2,6)
plot(nu_eNU,'k','Linewidth',1.5)
title('Policy Shock')
ylabel('Level')
grid on

saveas(gcf,'policy_shock_nk','epsc')
