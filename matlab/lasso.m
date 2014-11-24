%Not my own code.
% Function to run lasso by minimizing the following objective:
% f(beta) = 0.5*||y - X*beta||_2^2 + lambda*||beta||_1

% Inputs:
%   X: n-by-p matrix of input features
%   y: n-by-1 vector of output values
%   lambda: weight for l1 norm penalty

% Outputs:
%   beta: estimate of regression parameters

function beta = lasso(X,y,lambda)

% set options for optimization
miniter = 10;
maxiter = 4000;
tol = 1e-5;

% set fixed step size
t = 0.0001;

% initialize beta
p = size(X,2);
beta = zeros(p,1);

Grad = zeros(maxiter, p);

% optimize lasso objective 
for iter = 1:maxiter
    % compute gradient of f(beta) with respect to beta
    grad = (-(y - X*beta)'*(X))';
    Grad(iter, :) = grad;
    % compute new estimate of beta by appling soft thresholding operator
    beta_new = softthresh(beta-(t*grad),t*lambda);
     % compute new objective value
    obj_new = sum((y-X*beta_new).^2)/2 + lambda*sum(abs(beta_new));
    % check convergence
    if (iter >= miniter && abs(obj_new-obj) < tol)
        break;
    end
    % store new variables
    beta = beta_new;
    obj = obj_new;
end

%Grad
end

% Soft thresholding
function alpha = softthresh(beta,eta)
        
    beta(find(beta>eta)) = beta(find(beta>eta)) - eta;
    beta(find(beta<-eta)) = beta(find(beta<-eta)) + eta;
    beta(find(beta >= -eta & beta <= eta)) = 0;
    
    alpha = beta;
    %%% alpha = ?
    %%% Fill in computation of soft thresholding operator
    %%% Elements of beta whose values are larger than eta are decremented by eta (subtract eta)
    %%% Elements of beta whose values are smaller than -eta are incremented by eta (add eta)
    %%% Elements of beta whose value is in the range [-eta,eta] are set exactly to zero
    
end

