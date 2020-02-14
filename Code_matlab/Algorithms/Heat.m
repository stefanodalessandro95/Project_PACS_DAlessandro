function [x0,xmin,fmin,k,nepochs] = Heat(F,df,yy,a,b,mb,dim,plotFlag)
% HEAT STOCHASTIC GRADIENT DESCENT
% function F is given, depending only on x
% the gradient df depending on x and y (F is the sum of f over yy)
% F is minimized within the interval [a,b]
% the heat method is performed, starting from a random starting point x0, 
% picked from a uniform distribution on the interval [a,b]
% the stochastic gradient is computed by randomly picking a mini-batch of
% size mb from the set yy
% if plotFlag is set to 1, all the iterations are displayed on an image of
% the function, to show the behaviour of the algorithm

if dim == 1 % PARAMETERS 1D
    gamma0 = 0.0001;
    gamma1 = 0.00001; % parameters used to set gamma in the various iterations
    xstep_zero = 0.01; % starts from this value and is divided by the square root of the iteration number
    L = 20; % number of iterations on y before updating x
else % PARAMETERS 2D
    gamma0 = 0.05;
    gamma1 = 0.001; % parameters used to set gamma in the various iterations
    xstep_zero = 0.08; % starts from this value and is divided by the square root of the iteration number
    L = 20; % number of iterations on y before updating x
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
    title('Heat')
    pause(0.2)
elseif plotFlag && dim == 2
    plot3(x(1),x(2),F(x),'g*')
    pause(0.2)
end

k = 1;
% stopping criteria: max number of epochs, min change in loss function
while k < 200*N/(L*mb) && abs(F(x)-fprev) > 1.0e-4 
    gamma = gamma0*(1-gamma1)^(k/L); % update of gamma -> decreasing with iterations
    xstep = xstep_zero/sqrt(k*L*mb/N); % update of the step -> decreasing with the iterations
    
    grad_sum = zeros(dim,1);
    for i = 1:L  % Stochastic Gradient Langevin dynamics - computation of the average 
        grad_sum = grad_sum + (StGrad(x+sqrt(gamma)*randn(dim,1),df,yy,mb));
    end
    grad = grad_sum/L;
    
    % update of x
    fprev = F(x);
    x = x - xstep*grad;
    
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
nepochs =k*L*mb/N;

if plotFlag && dim == 1
    plot(xmin,fmin,'r*')
elseif plotFlag && dim == 2
    plot3(xmin(1),xmin(2),fmin,'r*')
end

end