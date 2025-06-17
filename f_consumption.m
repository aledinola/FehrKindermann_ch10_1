function c=f_consumption(aprime,a,z1,theta_i,agej,kappa_j,w,Jr,pension,r)
% The first three are the 'always required' next period
% endogenous states, this period endogenous states, exogenous states
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

if agej<Jr % If working age
    % kappa_j is age-deterministic profile
    % theta_i is individual permanent income type
    % z is persistent shock AR1
    c=w*kappa_j*theta_i*z1+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pension+(1+r)*a-aprime;
end


end %end function 
