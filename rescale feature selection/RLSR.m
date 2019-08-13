function [ ranked, theta,W,obj] = RLSR( XL, YL, XU, p,gamma, MAX_ITER,MAX_ITER2)
%LSFS Summary of this function goes here
%   Detailed explanation goes here
%   Input
%   -----
%   XL: {numpy array}, shape {n_features, n labeled samples}
%   YL: {numpy array}, shape {n labeled samples, n_clusters}
%   XU: {numpy array}, shape {n_features, n unlabeled samples}
%   p: norm parameter
%   gama: {float}, regular value
%   MAX_ITER£º maximum number of iterations in main loop
%   MAX_ITER£º maximum number of iterations for solving W
%   notes
%   -----
%   if bigger score the features get, the more importance features it is.
%   Output
%   ------
%  Pleaser cite this paper: Xiaojun Chen, Guowen Yuan, Feiping Nie, Joshua Zhexue Huang: Semi-supervised Feature Selection via Rescaled Linear Regression. IJCAI 2017: 1525-1531
%

epsilon=1e-5;
if nargin<4
    p=1;
end

if nargin<5
    gamma=1;
end

if nargin<6
    MAX_ITER=100;
end

if nargin<7
    MAX_ITER2=10;
end

q = 2/p - 1;

nl=size(XL,2);
num=nl+size(XU,2);
d=size(XL,1);
c=size(YL,2);
X=zeros(size(XL,1),num);
X(:,1:nl)=XL;
X(:,nl+1:num)=XU;

Y=ones(num,c)/c;
Y(1:nl,:)=YL;

H = eye(num) - ones(num)/num;
Q=eye(d)/d;

obj=zeros(MAX_ITER,1);

XHX=X*H*X';
for iter=1:MAX_ITER
    XHY=X*H*Y;
    obj_w_old=NaN;
    for iter2=1:MAX_ITER2
        W=(XHX+gamma*(Q^-2))\XHY;
        temp=sum(W.*W,2).^(1/(q+1))+epsilon;
        Q=diag( temp/ sum(temp) ).^(q/2);
        obj_w=F22norm(H*X'*W-H*Y)+gamma*trace(W'*Q*W);
        if iter2>1
            change=abs(obj_w_old-obj_w)/obj_w_old;
            if change<1e-7
                break;
            end
        end
        obj_w_old=obj_w;
    end
    b=(sum(Y,1)'-sum(W'*X,2))/num;
    
    % updata Yu
    for i=nl+1:num
        Y(i,:)=X(:,i)'*W+b';
        Y(i,:) = EProjSimplex_new(Y(i,:));
    end
    obj(iter)=F22norm(X'*W+repmat(b',[num 1])-Y)+gamma*L2Pnorm(W,p)^2;
    
    if iter ==1
        minObj=obj(iter);
        bestW = W;
    else
        if ~isnan(obj(iter)) && obj(iter) <= minObj
            minObj=obj(iter);
            bestW = W;
        end
    end
    
    if iter>1
        change=abs((obj(iter-1)-obj(iter))/obj(iter));
        if change<1e-8
            break;
        end
    end
end

W=bestW;

theta=sum(W.*W,2).^(p/2);
theta=theta/sum(theta);
[~, ranked] = sort(theta, 'descend');
end