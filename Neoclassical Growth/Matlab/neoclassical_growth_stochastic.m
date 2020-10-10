
%Author: Jacob Hess

%I wrote this code as a personal learning exercise and it is not optimized.
% The first part I solve the stochastic neoclassical growth model by discretized value
% function iteration. The stochastic variable TFP has three different states and 
% follows a markov process. The second part in order to make the savings policy function continuous
% I approximate it with chebyshev polynomials and minimize the residuals between
% the interpolants and the true function with ordinary least squares. I
% then simulate the tfp process and the model over 500 periods. 

%References that I used for this code are
%1) Chapter 5 in RBCs of ABCs by George McCandless
%2) Value Itertion Notes by Fabrice Collard (http://fabcol.free.fr/notes.html)
%3) Eric Sims Value Function Iteration Notes (https://www3.nd.edu/~esims1/grad_macro_17.html)
%4) Raul Santaeulalia-Llopis Function Approximation Notes (http://r-santaeulalia.net/Quantitative-Macro-F19-UnitI.html)


clear
close all
clc

tic

global v0 beta delta alpha kmat k0 prob a0 s j

plott=1; % set to 1 to see plots 

% set parameters. This model uses log utility.
alpha = 0.33; % capital share
beta = 0.95;
delta = 0.1; % depreciation rate (annual)
s=2; %sigma

%Steady State 
kgrid = 99; % grid points + 1
kstar = (alpha/(1/beta - (1-delta)))^(1/(1 - alpha)); % steady state k
cstar = kstar^(alpha) - delta*kstar;
istar = delta*kstar;
ystar = kstar^(alpha);
consgrowthstar=beta*(alpha*kstar.^(alpha-1) + (1-delta)).^(1/s);

%Stochastic Process
amat = [0.9 1 1.1]';
prob = 1/3*ones(3,3);

%Constructing grid and grid ranges
kmin = 0.25*kstar;
kmax = 1.75*kstar;
grid = (kmax-kmin)/kgrid;

kmat = kmin:grid:kmax;
kmat = kmat';
N = length(kmat);

polfun_cons = zeros(kgrid+1,length(amat));

tol = 0.01; %tolerance
maxits = 1000; %maximum number of allowed iterations
dif = tol+1000; 
its=1; %starting iteration value
v0=zeros(N,length(amat)); %initial guess



while dif>tol && its < maxits
    for j = 1:length(amat)
        for i = 1:N
        k0 = kmat(i,1);
        a0 = amat(j,1);
        k1 = fminbnd(@valfun_stoch,kmin,kmax);
        v1(i,j) = -valfun_stoch(k1);
        polfun_k1(i,j) = k1;
        end
    end
dif = norm(v1-v0);
v0 = v1;
its = its+1;
end

%loop for write the consumption policy function
for j = 1:length(amat)
    for i=1:N
    polfun_cons(i,j) = amat(j,1)*kmat(i,1)^(alpha) - polfun_k1(i,j) + (1-delta)*kmat(i,1);
    end
end

% Plotting value and policy functions
if plott==1 
figure;
plot(kmat,v1(:,1),'r',kmat,v1(:,2),'b',kmat,v1(:,3),'g','Linewidth',1)
title('Final Value Function')
xlabel('k_t')
ylabel('V(k)')
legend('A = 0.9','A = 1', 'A = 1.1', 'Location','Best')

figure;
plot(kmat,polfun_k1(:,1),'r',kmat,polfun_k1(:,2),'b',kmat,polfun_k1(:,3),'g','Linewidth',1)
title('k_{t+1} Policy Function')
xlabel('k_t')
ylabel('k_{t+1}')
legend('A = 0.9','A = 1', 'A = 1.1', 'Location','Best')

figure;
plot(kmat,polfun_cons(:,1),'r',kmat,polfun_cons(:,2),'b',kmat,polfun_cons(:,3),'g','Linewidth',1)
title('Consumption Policy function')
xlabel('k_t')
ylabel('c_t')
legend('A = 0.9','A = 1', 'A = 1.1', 'Location','Best')

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Chebyshev Approx %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k_init=0.75*kstar;  %initial capital
nrep=200;    %number of periods to simulate
k=zeros(nrep+1,1);  %initialize dynamics
y=zeros(nrep,1);        %some matrices
k(1)=k_init;



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





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% simulation %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Run the stochastic simulation of tfp. a is the state. id is the state
%index.
[a,id]=markov(prob,amat,nrep,1,123);



for t=1:nrep
    % Prepare the approximation for the capital stock
   trkt=2*(k(t)-kmin)/(kmax-kmin)-1;
   Tk			= [1 trkt];
   for i=3:n
      Tk=[Tk 2*trkt.*Tk(:,i-1)-Tk(:,i-2)];
   end
   
   k(t+1)=Tk*b(:,id(t));    % use the appropriate decision rule
y(t)=amat(id(t))*k(t).^alpha;  % computes output with the given tfp for that period.
   
end


i=k(2:nrep+1)-(1-delta)*k(1:nrep);
c=y-i;

figure;
subplot(221);plot(1:nrep,k(1:nrep));
set(gca,'fontname','times','fontsize',12);
xlabel('Time','fontname','times','fontsize',12);
title('Capital stock','fontname','times','fontsize',12);
subplot(222);plot(1:nrep,i);
set(gca,'fontname','times','fontsize',12);
xlabel('Time','fontname','times','fontsize',12);
title('Investment','fontname','times','fontsize',12);
subplot(223);plot(1:nrep,y);
set(gca,'fontname','times','fontsize',12);
xlabel('Time','fontname','times','fontsize',12);
title('Output','fontname','times','fontsize',12);
subplot(224);plot(1:nrep,c);
set(gca,'fontname','times','fontsize',12);
xlabel('Time','fontname','times','fontsize',12);
title('Consumption','fontname','times','fontsize',12);


toc

