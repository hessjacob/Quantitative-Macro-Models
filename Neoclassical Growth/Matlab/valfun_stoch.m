function val=valfun_stoch(k)

% This program gets the value function for a stochastic growth model with
% CRRA utility

global v0 beta delta alpha kmat k0 prob a0 s j

g = interp1(kmat,v0,k,'linear'); % smooths out previous value function

c = a0*k0^alpha - k + (1-delta)*k0; % consumption
if c <= 0
val = -8888888888888888-800*abs(c); % keeps it from going negative
else

val = (1/(1-s))*(c^(1-s) -1) + beta*(g*prob(j,:)');
end;

val = -val; % make it negative since we're maximizing and code is to minimize.