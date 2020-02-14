% F has to be a non-convex function wrt a parameter x (row), and it has to be
% like a loss function wrt a dataset yy (column)
clear all
close all
clc

% transformation function and its known gradient wrt the parameters
f1D = @(x,y) (sin(10*x).*cos(4*x.*y)-sin(6*y)).*exp(-5*(x-2).^2);
dfx1D = @(x,y) -10*(x-2).*f1D(x,y)+(10*cos(10*x).*cos(4*x.*y)...
    -4*y.*sin(10*x).*sin(4*x.*y)).*exp(-5*(x-2).^2);

% target function
g = @(y) f1D(2,y)+0.02*abs(y);

% random "training" data
D = 1000;
yy = [50*rand(D/2,1); (50*rand(D/2,1)+70)];
sort(yy);

% parameter space
K = 50000;
xx = linspace(1,3,K);

% quadratic loss function and its gradient wrt to the parameters
F = @(x) sum(0.5*(g(yy)-f1D(x,yy)).^2)/D;
dF = @(x) sum(-(g(yy)-f1D(x,yy)).*dfx1D(x,yy))/D;
ddf = @(x,y) -sum((g(y)-f1D(x,y)).*dfx1D(x,y))/length(y);

%% Gradient Descent
N = 1000;
glob_min = 0;
other_min = 0;
tic
for i=1:N
     [x0, xmin, Fmin, nit] = GD(F,dF,1.4,2.6,1,0);
    if Fmin<1.02
        glob_min = glob_min +1;
    elseif (xmin>=1.85 && xmin<=1.92) || (xmin>=2.17 && xmin<=2.23)
        other_min = other_min +1;
    end
    xmins_GD(i) = xmin;
end
mean_time_GD = toc/N;
percentage_GD = glob_min/N;
other_minimum_percentage_GD = other_min/N; 
figure()
plot(xx,F(xx))
xlim([1.4 2.6])
hold on
plot(xmins_GD,F(xmins_GD),'g*')
title('Results GD')

%% Stochastic Gradient Descent
N = 1000;
glob_min = 0;
other_min = 0;
tic
for i=1:N
    [x0, xmin, Fmin, nit] = SGD(F,ddf,yy,1.4,2.6,100,1,0);
    if Fmin<1.02
        glob_min = glob_min +1;
    elseif (xmin>=1.85 && xmin<=1.92) || (xmin>=2.17 && xmin<=2.23)
        other_min = other_min +1;
    end
    xmins_SGD(i) = xmin;
end
mean_time_SGD = toc/N;
percentage_SGD = glob_min/N;
other_minimum_percentage_SGD = other_min/N; 
figure()
plot(xx,F(xx))
xlim([1.4 2.6])
hold on
plot(xmins_SGD,F(xmins_SGD),'g*')
title('Results SGD')

%% Entropy Stochastic Gradient Descent
N =1000;
glob_min = 0;
other_min = 0;
tic
for i=1:N
    [x0,xmin,Fmin,k,nepochs] = EntropySGD(F,ddf,yy,1.4,2.6,100,1,0);
    if Fmin<1.02
        glob_min = glob_min +1;
    elseif (xmin>=1.85 && xmin<=1.92) || (xmin>=2.17 && xmin<=2.23)
        other_min = other_min +1;
    end
    xmins_EntropySGD(i) = xmin;
end
mean_time_EntropySGD = toc/N;
percentage_EntropySGD = glob_min/N; 
other_minimum_percentage_EntropySGD = other_min/N; 
figure()
plot(xx,F(xx))
xlim([1.4 2.6])
hold on
plot(xmins_EntropySGD,F(xmins_EntropySGD),'g*')
title('Results EntropySGD')

%% Heat
N = 1000;
glob_min = 0;
other_min = 0;
tic
for i=1:N
    [x0,xmin,Fmin,k,nepochs] = Heat(F,ddf,yy,1.4,2.6,100,1,0);
    if Fmin<1.02
        glob_min = glob_min +1;
    elseif (xmin>=1.85 && xmin<=1.92) || (xmin>=2.17 && xmin<=2.23)
        other_min = other_min +1;
    end
    xmins_Heat(i) = xmin;
end
mean_time_Heat = toc/N;
percentage_Heat = glob_min/N; 
other_minimum_percentage_Heat = other_min/N; 
figure()
plot(xx,F(xx))
xlim([1.4 2.6])
hold on
plot(xmins_Heat,F(xmins_Heat),'g*')
title('Results Heat')

