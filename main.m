%% Fehr and Kindermann chapter 10.1

clear,clc,close all
addpath(genpath('C:\Users\aledi\Documents\GitHub\VFIToolkit-matlab\VFIToolkit-matlab'))

% Options
vfoptions.verbose=1; % Just so we can see feedback on progress
simoptions=struct();
%simoptions.gridinterplayer=0;

% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d = 0; % Endogenous labour choice (fraction of time worked)
n_a = 1000; % Endogenous asset holdings
n_z = 7; % Exogenous labor productivity units shocks, persistent and transitiory
N_i = 2; % Permanent type of agents
N_j = Params.J; % Number of periods in finite horizon

%% The parameter that depends on the permanent type
% Fixed-effect (parameter that varies by permanent type)

sig2_theta = 0.242;
Params.theta_i=exp([-sqrt(sig2_theta),sqrt(sig2_theta)]);

PTypeDistParamNames={'dist_theta_i'};
Params.dist_theta_i=[0.5,0.5]; % Must sum to one
% Note: this is not relevant to solving the value function, but is needed for
% stationary distribition. It then gets encoded into the StationaryDist and
% so is not needed for things like life-cycle profiles.

%% Parameters

% Discount rate
Params.beta = 0.98;
% Preferences
Params.gamma = 2; % Coeff of relative risk aversion (curvature of consumption)

% Prices
Params.w=1; % Wage
Params.r=0.04; % Interest rate (0.05 is 5%)

% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=45;

% Age-dependent labor productivity units
Params.kappa_j = zeros(1,Params.J);
Params.kappa_j(1:Params.Jr-1)=[1.0000, 1.0719, 1.1438, 1.2158, 1.2842, 1.3527, ...
              1.4212, 1.4897, 1.5582, 1.6267, 1.6952, 1.7217, ...
              1.7438, 1.7748, 1.8014, 1.8279, 1.8545, 1.8810, ...
              1.9075, 1.9341, 1.9606, 1.9623, 1.9640, 1.9658, ...
              1.9675, 1.9692, 1.9709, 1.9726, 1.9743, 1.9760, ...
              1.9777, 1.9700, 1.9623, 1.9546, 1.9469, 1.9392, ...
              1.9315, 1.9238, 1.9161, 1.9084, 1.9007, 1.8354, ...
              1.7701, 1.7048];
Params.kappa_j(Params.Jr:Params.J) = 0;

% Replacement rate for pensions
Params.repl_pen=0.5;
% Pension benefit
Params.pension = Params.repl_pen*sum(Params.kappa_j)/(Params.Jr-1);

% persistent AR(1) process on idiosyncratic labor productivity units
Params.rho_z=0.985;
Params.sigma_epsilon_z=sqrt(0.022);

% Conditional survival probabilities: sj is the probability of surviving to be age j+1, given alive at age j
Params.sj = [1.00000, 0.99923, 0.99914, 0.99914, 0.99912, ...
                0.99906, 0.99908, 0.99906, 0.99907, 0.99901, ...
                0.99899, 0.99896, 0.99893, 0.99890, 0.99887, ...
                0.99886, 0.99878, 0.99871, 0.99862, 0.99853, ...
                0.99841, 0.99835, 0.99819, 0.99801, 0.99785, ...
                0.99757, 0.99735, 0.99701, 0.99676, 0.99650, ...
                0.99614, 0.99581, 0.99555, 0.99503, 0.99471, ...
                0.99435, 0.99393, 0.99343, 0.99294, 0.99237, ...
                0.99190, 0.99137, 0.99085, 0.99000, 0.98871, ...
                0.98871, 0.98721, 0.98612, 0.98462, 0.98376, ...
                0.98226, 0.98062, 0.97908, 0.97682, 0.97514, ...
                0.97250, 0.96925, 0.96710, 0.96330, 0.95965, ...
                0.95619, 0.95115, 0.94677, 0.93987, 0.93445, ...
                0.92717, 0.91872, 0.91006, 0.90036, 0.88744, ...
                0.87539, 0.85936, 0.84996, 0.82889, 0.81469, ...
                0.79705, 0.78081, 0.76174, 0.74195, 0.72155, ...
                0.00000];

%% Grids
% The ^3 means that there are more points near 0 and near 10. We know from
% theory that the value function will be more 'curved' near zero assets,
% and putting more points near curvature (where the derivative changes the most) increases accuracy of results.
a_min = 0;
a_max = 600;
a_grid=a_min+(a_max-a_min)*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

% First, the AR(1) process z
[z_grid,pi_z]=discretizeAR1_Rouwenhorst(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
z_grid=exp(z_grid); % Take exponential of the grid

% Grid for labour choice
d_grid=[];

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Notice: have added alpha_i to inputs (relative to Life-Cycle Model 11 which this extends)
ReturnFn=@(aprime,a,z,agej,theta_i,kappa_j,w,gamma,Jr,pension,r) ...
    f_ReturnFn(aprime,a,z,agej,theta_i,kappa_j,w,gamma,Jr,pension,r);

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
tic;

[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
toc

% V is now a structure containing the value function for each permanent type
V
% So for example for permanent type 1, V_i(a,z,j)
size(V.ptype001)
% Policy likewise depends on type, e.g.,
size(Policy.ptype001)
% which is the same as
[length(n_d)+length(n_a),n_a,n_z,N_j]

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,floor((n_z+1)/2))=1; % All agents start with zero assets, 
% and the median value of the shock

% Anything that is not made to depend on the permanent type is
% automatically assumed to be independent of the permanent type (that is,
% identical across permanent types). This includes things like the initial
% distribution, jequaloneDist

%% We now compute the 'stationary distribution' of households
% Start with a mass of one at initial age, use the conditional survival
% probabilities sj to calculate the mass of those who survive to next
% period, repeat. Once done for all ages, normalize to one
Params.mewj=ones(1,Params.J); % Marginal distribution of households over age
for jj=2:length(Params.mewj)
    Params.mewj(jj)=Params.sj(jj-1)*Params.mewj(jj-1);
end
Params.mewj=Params.mewj./sum(Params.mewj); % Normalize to one
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,pi_z,Params,simoptions);
% Again, we will explain in a later model what the stationary distribution
% is, it is not important for our current goal of graphing the life-cycle profile

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.earnings=@(aprime,a,z,w,kappa_j,theta_i) w*kappa_j*theta_i*z; 
FnsToEvaluate.assets=@(aprime,a,z) a; % a is the current asset holdings
FnsToEvaluate.consumption = @(aprime,a,z,agej,theta_i,kappa_j,w,Jr,pension,r) f_consumption(aprime,a,z,agej,theta_i,kappa_j,w,Jr,pension,r);

%% Calculate the life-cycle profiles
AgeStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, Params,n_d,n_a,n_z,N_j,N_i,d_grid, a_grid, z_grid, simoptions);

% By default, this includes both the 'grouped' statistics, like
% AgeConditionalStats.earnings.Mean
% Which are calculated across all permanent types of agents.
% And also the 'conditional on permanent type' statistics, like 
% AgeConditionalStats.earnings.ptype001.Mean

% Compute the coefficient of variation as in FK
coef_var.consumption = AgeStats.consumption.StdDeviation./AgeStats.consumption.Mean;
coef_var.earnings = AgeStats.earnings.StdDeviation./(max(AgeStats.earnings.Mean,1e-10));

%% Plot the life cycle profiles of earnings, both grouped and for each of the different permanent types

age_vec = Params.agejshifter+(1:1:Params.J);
figure
plot(age_vec,AgeStats.earnings.Mean)
hold on
plot(age_vec,AgeStats.consumption.Mean)
legend('Earnings','Consumption')
title('Life Cycle Profile: Labor Earnings')

figure
plot(age_vec,AgeStats.assets.Mean)
title('Life Cycle Profile: Assets')

figure
plot(age_vec,coef_var.earnings)
hold on
plot(age_vec,coef_var.consumption)
legend('Earnings','Consumption')
title('Coefficient of variation')
