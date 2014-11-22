% Function to run the Glasso algorithm:
% f(beta) = 0.5*||W11^(1/2)*beta - W11^(-1/2)*s12||_2^2 + lambda*||beta||_1

% Inputs:
%   X: n-by-p input matrix of n p-dimensional data points
%   lambda: weight for l1 norm penalty

% Outputs:
%   Theta: estimate (sparse) of the inverse covariance matrix


function [ W , T] = Glasso( X, lambda )

    %  Dimensions
    n = size(X,1);
    p = size(X,2); 

    % Normalize data to zero-mean distribution
    mu = sum(X,1)/n;
    X = X - repmat(mu, n, 1);

    % Initialize values
    S = (1/n)*((X')*(X));
    W = S + lambda*eye(p);
    Wnew = zeros(size(W));
    Beta = zeros(p, p);
    Theta = zeros(p, p);
    % set ierations.
    maxiter = 5000;
    tol = 1e-5;

    % Implement Glasso
    for iter = 1:maxiter
        for a = 1:p
            I = ~ismember(1:p, a);
            W11 = W(I, I);
            w22 = W(a,a);
            w12 = W(a,:);
            w12 = (w12(I))';
            S11 = S(I,I);
            s22 = S(a,a);
            s12 = S(a,:);
            s12 = (s12(I))';

            % Compute W11^(1/2)
            [V,D] = eig(W11);
            W11_sqrt = V * sqrt(D) * V';
            W11_sqrt_inv = inv(W11_sqrt);

            % Execute lasso
            b = (W11_sqrt_inv)*s12;
            beta = lasso(W11_sqrt, b, lambda);

            % Computer row/column of Wnew and Beta
            w12 = W11*beta;
            w12 = [w12(1:a-1); w22; w12(a:end)];
            Wnew(a,:) = w12';
            Wnew(:,a) = w12;
            Beta(a, :) = [beta(1:a-1); 0; beta(a:end)]';
            Beta(:, a) = [beta(1:a-1); 0; beta(a:end)];
        end

        diff = Wnew - W;
        if (abs(sum(diff(:))) < tol)
            break;
        end

        W = Wnew;

        for a = 1:p
            I = ~ismember(1:p, a);
            w22t = W(a,a);
            w12t = W(a,:);
            w12t = (w12t(I))';
            beta = Beta(a,:);
            beta = (beta(I))';

            theta22 = 1/(w22t - (w12t')*beta);
            theta12 = -beta*theta22;
            Theta(a, :) = [theta12(1:a-1); theta22; theta12(a:end)]';
            Theta(:, a) = [theta12(1:a-1); theta22; theta12(a:end)];
        end
        obj = log(det(Theta)) - trace(S*Theta) - lambda * (sum(sum(abs(Theta))))      
    end

    % Compute Theta 

    for a = 1:p
        I = ~ismember(1:p, a);
        w22t = W(a,a);
        w12t = W(a,:);
        w12t = (w12t(I))';
        beta = Beta(a,:);
        beta = (beta(I))';

        theta22 = 1/(w22t - (w12t')*beta);
        theta12 = -beta*theta22;
        Theta(a, :) = [theta12(1:a-1); theta22; theta12(a:end)]';
        Theta(:, a) = [theta12(1:a-1); theta22; theta12(a:end)];
    end
    
    
    % Plot binary image
    Plot(Theta);
    T=Theta;
end

