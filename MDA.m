function [acc_iter,pre,result] = MDA(s1,t1,Xs,Ys,Xt,Yt,options,result)

% Reference:
%% Jindong Wang, Wenjie Feng, Yiqiang Chen, Han Yu, Meiyu Huang, Philip S.
%% Yu. Visual Domain Adaptation with Manifold Embedded Distribution
%% Alignment. ACM Multimedia conference 2018.

%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim
%%% Yt      : Target domain label matrix, m * 1 (only used for testing accuracy)
%%% options : algorithm options:
%%%%% options.d      :  dimension after manifold feature learning (default: 20)
%%%%% options.T      :  number of iteration (default: 10)
%%%%% options.lambda :  lambda in the paper (default: 10)
%%%%% options.eta    :  eta in the paper (default: 0.1)
%%%%% options.rho    :  rho in the paper (default: 1.0)
%%%%% options.base   :  base classifier for soft labels (default: NN)

%% Outputs:
%%%% Acc      :  Final accuracy value
%%%% acc_iter :  Accuracy value list of all iterations, T * 1
%%%% Beta     :  Cofficient matrix
%%%% Yt_pred  :  Prediction labels for target domain

%% Algorithm starts here
%Xt=train_data;


% Manifold feature learning
[Xs_new,Xt_new,~] = GFK_Map(Xs,Xt,options.d);
Xs = double(Xs_new');
Xt = double(Xt_new');
%    %baseline
%     Xs = Xs';
%     Xt = Xt';
%    %
X = [Xs,Xt];
n = size(Xs,2);
m = size(Xt,2);
C = length(unique(Ys));
acc_iter = [];
%
%     YY = [];
%     for c = 1 : C
%         YY = [YY,Ys==c];
%     end
%     YY = [YY;zeros(m,C)];

%% Data normalization
X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));

%% Construct graph Laplacian
if options.rho > 0
    manifold.k = options.p;
    manifold.Metric = 'Cosine';
    manifold.NeighborMode = 'KNN';
    manifold.WeightMode = 'Cosine';
    W = lapgraph(X',manifold);
    Dw = diag(sparse(sqrt(1 ./ sum(W))));
    L = eye(n + m) - Dw * W * Dw;
else
    L = 0;
end

% Generate soft labels for the target domain
knn_model = fitcknn(X(:,1:n)',Ys,'NumNeighbors',1);
y_pseudo= knn_model.predict(X(:,n + 1:end)');




% Construct MMD matrix
e = [1 / n * ones(n,1); -1 / m * ones(m,1)];
M0 = e * e' * length(unique(Ys));

N=0;
for c = reshape(unique(Ys),1,C)
    e = zeros(m+n,1);
 
    
    e(Ys==c) = 1 / length(find(Ys==c));
    e(n+find(y_pseudo==c)) = -1 / length(find(y_pseudo==c));
    e(isinf(e)) = 0;
    N = N + e*e';
end

M = (1 - options.mu) * M0 + options.mu * N;
M = M / norm(M,'fro');
H=eye(n+m) - 1/n * ones(n+m,n+m);
% Compute coefficients vector Beta
[A,~]=eigs(X*M*X'+ X*options.rho * L*X'+options.lambda*eye(size(X,1)),X*X',options.dim,'SM');
Z=A'*X;
%         [A,~] = eigs(K*M*K'+ options.rho * L*K'+options.lambda*eye(m+n),K*K',options.dim,'SM');
%             Z = A'*K;


%normalization for better classification performance
%
%         Zs = Z(:,1:n)';
%         Zt = Z(:,n+1:end)';
%          Zs=Zs*diag(sparse(1./sqrt(sum(Zs.^2))));
%        Zt=Zt*diag(sparse(1./sqrt(sum(Zt.^2))));
Zs=A'*Xs;Zs=Zs';
Zt=A'*Xt;Zt=Zt';
Zs=Zs*diag(sparse(1./sqrt(sum(Zs.^2))));
Zt=Zt*diag(sparse(1./sqrt(sum(Zt.^2))));


        model = train(Ys,sparse(real(Zs)),'-s 0 -c 1 -B -1 -q');
        [pre,~, prob_estimates] = predict(Yt, sparse(real(Zt)), model,'-b 1');
        
                if (pre(1)-1 == 0) && (prob_estimates(1,1)< prob_estimates(1,2))
                   score = prob_estimates(:,1);
                elseif (pre(1)-1 == 0) && (prob_estimates(1,1)> prob_estimates(1,2))
                    score = prob_estimates(:,2);
                elseif (pre(1)-1 == 1) && (prob_estimates(1,1)< prob_estimates(1,2))
                    score = prob_estimates(:,2);
                elseif (pre(1)-1 == 1) && (prob_estimates(1,1)> prob_estimates(1,2))
                    score = prob_estimates(:,1);
                end
              

   
        %
        mea = performanceMeasure(s1,t1,Yt-1,score-1,pre-1);
        result= [result;mea];
        mea
  
        
        
        
end
