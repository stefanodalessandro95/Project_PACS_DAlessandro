function [grad] = StGrad(x,df,yy,mb)
% Given a gradient function, depending on a parameter x and a value y
% the stochastic gradient at point x is performed by averaging tha gradient
% computed on a mini-batch of size mb, by randomly picking mb samples from
% the vector yy

N = length(yy);
batch = randi([1,N],mb,1);

grad = df(x,yy(batch));


end

