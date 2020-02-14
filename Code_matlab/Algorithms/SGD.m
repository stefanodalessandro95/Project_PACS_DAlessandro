function [x0,xmin,fmin,nit] = SGD(F,df,yy,a,b,mb,dim,plotFlag)
% STOCHASTIC GRADIENT DESCENT 
% function F is given, depending only on x
% the gradient df depending on x and y (F is the sum of f over yy)
% F is minimized within the interval [a,b]
% the stochastic gradient descent method is performed, starting from a
% random starting point x0, picked from a uniform distribution on the
% interval [a,b]
% the stochastic gradient is computed by randomly picking a mini-batch of
% size mb from the set yy
% if plotFlag is set to 1, all the iterations are displayed on an image of
% the function, to show the behaviour of the algorithm

% starting point is randomly taken in uniform way in the domain 
N = length(yy);
x0 = a+(b-a)*rand(dim,1);

beta = 0.8;
x = x0;
fprev = F(x)+1;
t0 = (b-a)/8;

t = t0;
nit = 0;

if plotFlag && dim == 1
    xx = linspace(a,b,1000);
    figure()
    hold on
    plot(xx,F(xx))
    plot(x,F(x),'g*')
    title('SGD')
end

% stopping criteria: min change in loss function, min length of forward
% step
while t > 1.0e-6 && abs(F(x)-fprev) > 1.0e-5 && nit < 500  
    nit = nit + 1;
    t = t0^(1+nit*mb/N);
    g = -t*StGrad(x,df,yy,mb);
    
    while (F(x+g)>=F(x) || norm(g)>2) && norm(g)>10e-6 
        t = beta*t;
        g = -t*StGrad(x,df,yy,mb);
        if t<1.0e-6
            g = zeros(dim,1);
        end
    end
    
    fprev = F(x);
    % update of x
    x = x + g;
    
    if plotFlag && dim == 1
        plot(x,F(x),'g*');
        pause(0.2);
    elseif plotFlag && dim == 2
        plot3(x(1),x(2),F(x),'g*');
        pause(0.2)
    end

end

xmin = x;
fmin = F(x);

if plotFlag && dim == 1
    plot(xmin,fmin,'r*');
elseif plotFlag && dim == 2
    plot3(xmin(1),xmin(2),fmin,'r*');
end

end

