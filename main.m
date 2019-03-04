addpath(genpath('bads-master'));
result=[];
%% Input arguments: 
%                  [x,fval,exitflag,output] = bads(@fun,x0,lb,ub,plb,pub);
%                                         x0:Initialization parameters
%                                         lb:Upper bound of parameter search
%                                         ub:Low bound of parameter search
%                                         plb:Plausible upper bound of parameter search
%                                         pub::Plausible low bound of parameter search
% Output arguments: 
%                   x:Optimized parameters
%                   fval:Classification error rate of feature subset
%                   exitflag:Characteristics of termination iteration
%                   output:returns a structure OUTPUT
%% Coding information
% Editor:Xuesen Yang
% Institution: Shenzhen University
% E-mail:1348825332@qq.com
% Edit date:2019-3-4 
%% Sample
for r=1:30
[x,fval,exitflag,output] = bads(@FeatureRank,[0.1 0.2 5],[0 0 5],[1 1 30],[0 0 5],[1 1 10]);
result=[result;x,fval]
end
result(:,3)=round(result(:,3));
save('FRank','result')