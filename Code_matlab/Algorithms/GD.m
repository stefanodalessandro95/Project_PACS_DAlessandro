function [x0, xmin, fmin, nit] = GD(f,df,a,b,dim,plotFlag)
% GRADIENT DESCENT 
% function f is given, along with its gradient df and an interval [a,b] in
% which to look for the minimum
% step size is computed with backtraking line search 
% if plotFlag is set on 1, alle the iterations are shown, also with the
% backtraking procedure

% parameters for the backtracking
alpha = 0.3;
beta = 0.8;

% starting point is randomly taken in uniform way in the domain 
x0 = a+(b-a)*rand(dim,1);

x = x0;
fprev = f(x)+1;
t0 = (b-a)/3;
t = t0;
nit = 0;

if plotFlag
    xx = linspace(a,b,1000);
end

% stopping criteria: min change in the loss function, min change in the
% forward step t 
while t > 1.0e-6 && abs(f(x)-fprev) > 1.0e-6  
    nit = nit + 1;
    g = -df(x)/norm(df(x));
    t = t0;
    
    if plotFlag && dim<3
        if dim == 1
            close all
            figure(); hold on
            plot(xx,f(xx))
            plot(x,f(x),'r*')
            plot(x+t*g, f(x+t*g),'g*')
            plot(x+t*g, f(x)+alpha*t*(df(x)')*g, 'k*')
            title('GD')
        elseif dim == 2
            plot3(x(1),x(2),f(x),'r*')
            plot3(x(1)+t*g(1),x(2)+t*g(2), f(x+t*g),'g*')
            plot3(x(1)+t*g(1),x(2)+t*g(2), f(x)+alpha*t*(df(x)')*g, 'k*')
        end
        pause(0.2);
    elseif plotFlag 
        disp('Dimention too high for plotting')
    end
    
    % backtracking line search 
    while f(x+t*g) > f(x) + alpha*t*(df(x)')*g
          t = beta*t;
          
          if plotFlag && dim == 1
              plot(x+t*g, f(x+t*g),'g*')
              plot(x+t*g, f(x) + alpha*t*(df(x)')*g, 'k*')
              pause(0.2)
          elseif plotFlag && dim == 2
              plot3(x(1)+t*g(1),x(2)+t*g(2), f(x+t*g),'g*')
              plot3(x(1)+t*g(1),x(2)+t*g(2),f(x) + alpha*t*(df(x)')*g, 'k*')
              pause(0.2)
          end

    end
    fprev = f(x);
    
    % update of x 
    x = x + t*g;
end

xmin = x;
fmin = f(x);

if plotFlag && dim == 1
    plot(xmin,fmin,'r*');
elseif plotFlag && dim == 2
    plot3(xmin(1),xmin(2),fmin,'r*');
end

end

