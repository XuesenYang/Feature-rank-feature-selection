function [fitness] = FeatureRank(x)
% Function that implements the FeatureRank algorithm.
% Input arguments: 
%                  H---correlation matrix 
%                  alpha---probability factor that describes  how likely a feature switches another one
%                  deta---Reset the value of correlation coefficient <deta  to 0
% Output arguments: 
%                 fitness----Classification error rate of feature subset
% Editor:Xuesen Yang
% Institution: Shenzhen University
% E-mail:1348825332@qq.com
% Edit date:2019-3-4 
%% Three parameters to be optimized
alpha=x(1);
deta=x(2);
fsize=round(x(3));
%% load dataset
load('LSVT_voice_rehabilitation.mat')
X=data(:,1:end-1);
Y=data(:,end);
group=Y;
class=unique(Y);
H=abs(corr(X));
change_element=H<deta;
H(change_element)=0; % remove edges which weighs less than ¦È
% H1=abs(corr(X,Y));
% alpha=0.2;
n = length(H);
a = sparse(n,1);           
e = ones(n,1);
a(find(all(H==0,2)))=1;             % vector keeping track of non-zero rows in H
b=diag(1./sum(H,2))*H;
H = sparse(diag(1./sum(H,2))*H);    % weight each link

iter = 0;
ranks = e'/n;                       % initiate ranks vector
oldrank = zeros(1,n);

%% iterate until convergence
while(norm(ranks-oldrank) > 1e-7)
    iter = iter + 1;
    oldrank = ranks;
    A=(alpha*(H + a*(e'/n)) + ((1-alpha)*e*(e'/n)));
    ranks = oldrank*(alpha*(H + a*(e'/n)) + ((1-alpha)*e*(e'/n)));
    % the big expression within paranthesis that is multiplied with oldrank
    % is the google matrix G 
end
[fg,w]=sort(ranks,'descend');
Selected_feature=w(1:fsize);
e=10;   % e-fold CrossValidation
%% caculate the classification error rate of feature subset
for i=1:length(class)
    sa=[];
    sa=data((group==class(i)),:);
    [number_of_smile_samples,~] = size(sa); % Column-observation
    smile_subsample_segments1 = round(linspace(1,number_of_smile_samples,e+1)); % indices of subsample segmentation points    
    data_group{i}=sa;
    smile_subsample_segments{i}=smile_subsample_segments1;
end
for i=1:e   
    data_ts=[];data_tr =[];
    for j=1:length(class)
      smile_subsample_segments1=smile_subsample_segments{j};
      sa=data_group{j};
      test= sa(smile_subsample_segments1(i):smile_subsample_segments1(i+1) , :); % current_test_smiles
      data_ts=[test;data_ts] ; % Test data
      train = sa;
      train(smile_subsample_segments1(i):smile_subsample_segments1(i+1),:) = [];
      data_tr =[train;data_tr];% Training data
    end
    mdl = fitcknn(data_tr(:,Selected_feature),data_tr(:,end),'NumNeighbors',4,'Standardize',1);% K nearest neighbor classifier
    Ac1=predict(mdl,data_ts(:,Selected_feature)); 
    Fit(i)=sum(Ac1~=data_ts(:,end))/size(data_ts,1);
end
    fitness=mean(Fit); % classification error rate
end


