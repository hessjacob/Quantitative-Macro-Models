function [chain,state]=markov(PI,s,n,s0,seed);
%
% [chain,state]=markov(PI,s,n,s0); 
%
% simulate a Markov chain
%
% PI	: Transition matrix
% s	: State vector
% n	: length of simulation
% s0	: initial state (index)
%
% chain : values for the simulated markov chain
% state : index of the state
%
[rpi,cpi]=size(PI);
s=s(:);
if ~(rpi==cpi);
   error('Transition matrix must be square')
end
if ~(size(s,1)==cpi);
   error('Number of state does not match size of Transition matrix')
end

cum_PI=[zeros(rpi,1) cumsum(PI')'];

if nargin<4;
   s0=1;
end

if nargin>4;
   rand('state',seed)
end
sim		= rand(n,1);
state		= zeros(n,1);
state(1) 	= s0;
%
for k=2:n;
  state(k)=find(((sim(k)<=cum_PI(state(k-1),2:cpi+1))&(sim(k)>cum_PI(state(k-1),1:cpi)))); %the state is determined if less than 1) the cumulative sum of the state and 2) if it is greater than the cumulative sum of 0 state1, state2. will take the state which both have ones. 
end;
chain=s(state);