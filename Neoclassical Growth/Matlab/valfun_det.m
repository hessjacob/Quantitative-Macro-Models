function val=valfun_det(k) 

% This program gets the value function for a neoclassical growth model with
% no uncertainty and CRRA utility

global v0 beta delta alpha kmat k0 s 

g = interp1(kmat,v0,k,'linear'); % smooths out previous value function 

c = k0^alpha - k + (1-delta)*k0; % consumption (This is the constraint)

if c<0
    val = -888888888888888888-800*abs(c); % keeps it from going negative
else
     
val=(1/(1-s))*((c)^(1-s)-1) + beta*g; %log utility approximately will produce the same results as s=1.01

end
val = -val; % make it negative since we're maximizing and code is to minimize.