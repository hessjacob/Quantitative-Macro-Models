
%Neoclassical Growth Model -- Deterministic Simulation

%Author: Jacob Hess

%I wrote this code as a personal learning exercise and it is not optimized.
% The first part solve the deterministic neoclassical growth model by discretized value
% function iteration. The second part in order to make the savings policy function continuous
% I approximate it with chebyshev polynomials and minimize the residuals between
% the interpolants and the true function with ordinary least squares. I
% then simulate the model by setting initial capital below the steady state
% and plot the transition 

%To make the policy function continuous you can choose to use cubic spline
%or chebshev interpolation. Chebyshev will work a lot better. 

%References that I used for this code are
%1) Chapter 5 in RBCs of ABCs by George McCandless
%2) Value Itertion Notes by Fabrice Collard (http://fabcol.free.fr/notes.html)
%3) Eric Sims Value Function Iteration Notes (https://www3.nd.edu/~esims1/grad_macro_17.html)
%4) Raul Santaeulalia-Llopis Function Approximation Notes (http://r-santaeulalia.net/Quantitative-Macro-F19-UnitI.html)

clear
close all
clc
set(0, 'DefaultLineLineWidth', 1.5);

tic

global v0 beta delta alpha kmat k0 s

plott=1; % set to 1 to see plots 
approx = 2; %set 1 for cubic spline. set 2 for chebyshev. 

% set parameters. This model uses log utility.
alpha = 0.33; % capital's share
beta = 0.95;
delta = 0.1; % depreciation rate (annual)
s=1.5; %sigma

%Steady State 
kgrid = 99; % grid points + 1
kstar = (alpha/(1/beta - (1-delta)))^(1/(1 - alpha)); % steady state k
cstar = kstar^(alpha) - delta*kstar;
istar = delta*kstar;
ystar = kstar^(alpha);
consgrowthstar=beta*(alpha*kstar.^(alpha-1) + (1-delta)).^(1/s);

%Constructing grid and grid ranges
kmin = 0.25*kstar;
kmax = 1.75*kstar;
grid = (kmax-kmin)/kgrid;

kmat = kmin:grid:kmax;
kmat = kmat';
N = length(kmat);

polfun_k1 = zeros(kgrid+1,1);

tol = 0.01; %tolerance
maxits = 1000; %maximum number of allowed iterations
dif = tol+1000; 
its=1; %starting iteration value
v0=zeros(N,1); %initial guess

while dif>tol && its < maxits
for i=1:N
k0 = kmat(i,1); %Takes each value i in the state space from the grid
k1 = fminbnd(@valfun_det,kmin,kmax); %Arg max of bellman equation (in function file). Fminbind finds the minimum on the specific interval. Our interval is the grid limits.
v1(i,1) = - valfun_det(k1); %Takes new optimized values for i and plugs it into the new function
polfun_k1(i,1) = k1; %Finds the policy function associated with k1
end
V_store(:,its) = v1; %storing the value functions to store value functions over the iterations. This is to plot the convergence
dif = norm(v1-v0); %measure the euclidean distance between value functions
v0=v1; % if dif>tol then v1 because the v0 for the function file to do the iteration again
its= its+1; %iteration counter
end

%loop for write the consumption policy function
for i=1:N
polfun_cons(i,1) = kmat(i,1)^(alpha) - polfun_k1(i,1) + (1-delta)*kmat(i,1);
end

%Plotting value and policy functions
if plott==1 
figure
plot(kmat,v1,'k','Linewidth',1)
title('Final Value Function')
xlabel('k')
ylabel('V(k)')

figure 
plot(kmat,polfun_k1,'k','Linewidth',1)
title('k_{t+1} Policy Function')
xlabel('k_t')
ylabel('k_{t+1}')
ssline = refline([1 0]);
ssline.LineStyle = '--';
legend('Policy Function','45 degree line','Location','Best')

figure
plot(kmat,polfun_cons,'k','Linewidth',1)
title('Consumption Policy function')
xlabel('k')
ylabel('c')

%plot the value function convergence
figure
hold on
 for k = 1:9:its
        txt = ['Iteration ',num2str(k)];
        plot(kmat,V_store(:,k),'DisplayName',txt)
 end
legend show
set(legend,'location','northwest')
xlabel('k_t')
ylabel('v(k)')
title('Value Function Iterations')
hold off

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Transition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k_init=0.5*kstar;  %initial capital
nrep=50;    %number of periods to simulate
k=zeros(nrep+1,1);  %initialize dynamics
k(1)=k_init;

if approx == 1 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% cubic spline approx %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


k(1)=k_init;
for t=1:nrep
   k(t+1)=interp1(kmat,polfun_k1,k(t),'pchip'); %pchip means cubic interpolation
end

end

if approx == 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Chebyshev approx %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%We are working to get the decision rule of form: k' = sum_{l=0} ^n
%\theta_l T_l(\gamma(k))
    
%step 1: Apply this transformation to the data points. This is \gamma(k).
%We are transforming the grid points from [kmin to kmax] to [-1 to 1]. 


transk	= 2*(kmat-kmin)/(kmax-kmin)-1; % EVENLY SPACED nodes. 

% Computes the matrix of Chebychev polynomials

n			= 10;    %order approximaztion
Tk			= [ones(kgrid+1,1) transk]; %Chebyshev polynomials of order 0 and 1. 
%We know that a chebyshev polynomial of order 0 is 1 (column vector of ones). Order 1 is the nodes themselves.  

% Now we carry out the higher order polynomials and store them in their respective column. 
% This is the recursive scheme from slide 42 of Raul Santaeulalia's
% Function Approxmiations.

for i=3:n
   Tk=[Tk 2*transk.*Tk(:,i-1)-Tk(:,i-2)];
end

%b=Tk\polfun_k1;        %computes OLS
b = inv((Tk'*Tk))*Tk'*polfun_k1; %computes OLS



%plot the discrete (every fifth grid point) vs. the approximated continuous
%function
plot(kmat(1:5:100),polfun_k1(1:5:100),'o',kmat,Tk*b,'-');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% simulation %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for t=1:nrep
   trkt=2*(k(t)-kmin)/(kmax-kmin)-1;
   Tk			= [1 trkt];
    for i=3:n
      Tk=[Tk 2*trkt.*Tk(:,i-1)-Tk(:,i-2)];
    end
   k(t+1)=Tk*b;
end
    
end

y=k(1:nrep).^alpha;
i=k(2:nrep+1)-(1-delta)*k(1:nrep);
c=y-i;



figure;
subplot(221);plot(1:nrep,k(1:nrep),1:nrep,kstar*ones(nrep,1),'--');
set(gca,'fontname','times','fontsize',12);
xlabel('Time','fontname','times','fontsize',12);
title('Capital stock','fontname','times','fontsize',12);
subplot(222);plot(1:nrep,c,1:nrep,cstar*ones(nrep,1),'--');
set(gca,'fontname','times','fontsize',12);
xlabel('Time','fontname','times','fontsize',12);
title('Consumption','fontname','times','fontsize',12);
subplot(223);plot(1:nrep,y,1:nrep,ystar*ones(nrep,1),'--');
set(gca,'fontname','times','fontsize',12);
xlabel('Time','fontname','times','fontsize',12);
title('Output','fontname','times','fontsize',12);
subplot(224);plot(1:nrep,i,1:nrep,istar*ones(nrep,1),'--');
set(gca,'fontname','times','fontsize',12);
xlabel('Time','fontname','times','fontsize',12);
title('Investment','fontname','times','fontsize',12);

toc
