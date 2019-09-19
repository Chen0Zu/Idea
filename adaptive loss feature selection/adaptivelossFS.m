function [W, b, obj] = adaptivelossFS(X, y, sigma, lambda)
% min_{W,b}  ||X*W+1*b'-Y||_sigma+lambda*||W||_2,1
% that is: min_{W,b}  \sum_i (1+sigma)*||xi*W+b'-yi||_2^2/(||xi*W+b'-yi||_2+sigma)
% Input:
%   X: n*d data matrix, each row is an sample point
%   Y: n*c label matrix, each row is an sample label
%   sigma: parameter of adaptive loss function
%   lambda: regularization parameter
% Output:
%   W: d*c coefficient matrix
%   b: c*1 bias vector
%   obj: objective values
max_iter = 50;
obj = zeros(max_iter, 1);
[n,d] = size(X);
c = length(unique(y));
% one hot encoding
I = eye(max(y));
Y = I(y,:);
% initialization
W = rand(d,c);

% iteration
for i = 1:max_iter
    
    E = X*W+ones(n,1)*b'-Y;
    e = sqrt(sum(E.^2,2));
    d = (1+sigma)/2*(e+2*sigma)./(e+sigma).^2;
    D = spdiags(d,0,n,n);
    
    obj(i) = sum((1+sigma)*e.^2/(e+sigma)) + lambda*sqrt(sum(W.^2,2));
    
    d_hat = 1./sqrt(sum(W.^2, 2));
    D_hat = spdiags(d_hat, 0, d, d);
    
    N = D - 1/sum(d)*d*d';
    W = inv(X'*N*X+lambda*D_hat)*(X'*N*Y);
    b = 1/sum(d)*(Y'*d-W'*X'*d);
end