function [x0,xmin,fmin,k,nepochs] = EntropySGD(F,df,yy,a,b,mb,dim,plotFlag)
% ENTHROPY STOCHASTIC GRADIENT DESCENT 
% function F is given, depending only on x
% the gradient df depending on x and y (F is the sum of f over yy)
% F is minimized within the interval [a,b]
% the enthropy stochastic gradient descent method is performed, starting from a
% random starting point x0, picked from a uniform distribution on the
% interval [a,b]
% the stochastic gradient is computed by randomly picking a mini-batch of
% size mb from the set yy
% if plotFlag is set to 1, all the iterations are displayed on an image of
% the function, to show the behaviour of the algorithm

if dim == 1 % PARAMETERS 1D
    beta = 800;
    gamma0 = 0.2;
    gamma1 = 0.01; 
    ystep = 0.01;  % fixed 
    xstep = 0.005; % starts from this value and is reduced by a factor 5 after 3 epochs 
    alpha = 0.8; % forwarding average for y
    L = 15; % number of iterations on y before updating x
    rho = 0.9; 
    
else % PARAMETERS 2D
    beta = 100;
    gamma0 = 1;
    gamma1 = 0.1;
    ystep = 0.04; % fixed
    xstep = 0.1; % starts from this value and is reduced by a factor 5 after 3 epochs 
    alpha = 0.9; % forwarding average for y
    L = 10; % number of iterations on y before updating x
    rho = 0.01;
end

% dimention of the dataset is computed for termination conditions
N = length(yy);

% starting point is randomly taken in uniform way in the domain
x0 = a+(b-a)*rand(dim,1);
x = x0;
fprev = F(x)+1;

if plotFlag && dim == 1
    xx = linspace(a,b,1000);
    figure()
    hold on
    plot(xx,F(xx))
    plot(x,F(x),'g*')
    title('Entropy SGD')
    pause(0.2)
elseif plotFlag && dim == 2
    plot3(x(1),x(2),F(x),'g*')
    pause(0.2)
end

k = 1;
% stopping criteria: maximum epochs, min change in loss function
while k < 200*N/((L+1)*mb) && abs(F(x)-fprev) > 1.0e-4
    gamma = gamma0*(1-gamma1)^(k/L); % update of gamma -> decreasing
    if mod(k*(L+1)*mb,3*N)==0 
        xstep = xstep/5; % update of forward step, stepwise decreasing after 3 epochs
    end
    y = x;
    y_mean = x;
    
    for i = 1:L  % Stochastic gradient Langevin dymanics - computation of the average 
        grad = StGrad(y,df,yy,mb);
        y = y - ystep*(grad+(y-x)/gamma)+sqrt(ystep/beta)*randn(dim,1);
        y_mean = (1-alpha)*y_mean+alpha*y; 
    end
    
    grad = StGrad(x,df,yy,mb);
    fprev = F(x);
    % update of x
    x = x - xstep*(rho*grad+(x-y_mean)/gamma);
    
    if plotFlag && dim == 1
        plot(x,F(x),'g*');
        pause(0.2)
    elseif plotFlag && dim == 2
        plot3(x(1),x(2),F(x),'g*');
        pause(0.2)
    end

    k = k+1;
end

xmin = x;
fmin = F(x);
nepochs =k*(L+1)*mb/N;

if plotFlag && dim == 1
    plot(xmin,fmin,'r*')
elseif plotFlag && dim == 2
    plot3(xmin(1),xmin(2),fmin,'r*')
end

end
