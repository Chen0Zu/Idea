function [W, b, obj] = adaptivelossFS(X, Y, lambda)
% min_{W,b}  ||X*W+1*b'-Y||_sigma+lambda*||W||_2,1
% that is: min_{W,b}  \sum_i (1+sigma)*||xi*W+b'-yi||_2^2/(||xi*W+b'-yi||_2+sigma)
% Input:
%   X: n*d data matrix, each row is an sample point
%   Y: n*c label matrix, each row is an sample label
%   lambda: regularization parameter
% Output:
%   W: d*c coefficient matrix
%   b: c*1 bias vector
%   obj: objective values

end