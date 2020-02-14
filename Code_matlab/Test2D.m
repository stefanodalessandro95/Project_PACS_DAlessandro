% F has to be a non-convex function wrt a parameter x (row), and it has to be
% like a loss function wrt a dataset yy (column)
clear all
close all
clc

% target function
g = @(y) f2D([2;2],y);

% random "training" data
D = 1000;
yy = 20*rand(D,1);
yy = sort(yy);

% parameter space
K = 100;
xx(1,:) = repmat(linspace(1,3,K),1,K);
xx(2,:) = sort(repmat(linspace(1,3,K),1,K));

% quadratic loss function and its gradient wrt to the parameters
F = @(x) sum(0.5*(g(yy)-f2D(x,yy)).^2)/D;
dF = @(x) -dfx2D(x,yy)*(g(yy)-f2D(x,yy))/D;
ddf = @(x,y) -dfx2D(x,y)*(g(y)-f2D(x,y))/length(y);

%% Gradient Descent (exact)
zz = F(xx);
zz = reshape(zz,[K,K])';
figure(1)
surf(linspace(1,3,K),linspace(1,3,K),zz)
hold on 
title('GD')
hold on
N = 1000;
glob_min = 0;
tic
for i=1:N
    [x0, xmin, Fmin, nit] = GD(F,dF,1,3,2,0);
    if Fmin<0.01
        glob_min = glob_min +1;
    end
    mins_GD(:,i) = xmin;
end
mean_time_GD = toc/N;
percentage_GD = glob_min/N;
plot3(mins_GD(1,:),mins_GD(2,:),F(mins_GD),'g*') 

%% Stochastic Gradient Descent
zz = F(xx);
zz = reshape(zz,[K,K])';
figure(2)
surf(linspace(1,3,K),linspace(1,3,K),zz)
title('SGD')
hold on
N = 1000;
glob_min = 0;
tic
for i=1:N
    [x0, xmin, Fmin, nit] = SGD(F,ddf,yy,1,3,100,2,0);
    if Fmin<0.01
        glob_min = glob_min +1;
    end
    mins_SGD(:,i) = xmin;
end
mean_time_SGD = toc/N;
percentage_SGD = glob_min/N; 
plot3(mins_SGD(1,:),mins_SGD(2,:),F(mins_SGD),'g*') 

%% Entropy Stochastic Gradient Descent
zz = F(xx);
zz = reshape(zz,[K,K])';
figure(3)
surf(linspace(1,3,K),linspace(1,3,K),zz)
title('Entropy SGD')
hold on
N = 1000;
glob_min = 0;
tic
for i=1:N
    [x0,xmin,Fmin,k,nepochs] = EntropySGD(F,ddf,yy,1.4,2.6,100,2,0);
    if Fmin<0.01
        glob_min = glob_min +1;
    end
    mins_EntropySGD(:,i) = xmin;
end
mean_time_EntropySGD = toc/N;
percentage_EntropySGD = glob_min/N; 
plot3(mins_EntropySGD(1,:),mins_EntropySGD(2,:),F(mins_EntropySGD),'g*')

%% Heat
zz = F(xx);
zz = reshape(zz,[K,K])';
figure(4)
surf(linspace(1,3,K),linspace(1,3,K),zz)
title('Heat')
hold on
N = 1000;
glob_min = 0;
tic
for i=1:N
    [x0,xmin,Fmin,k,nepochs] = Heat(F,ddf,yy,1.4,2.6,100,2,0);
    if Fmin<0.01
        glob_min = glob_min +1;
    end
    mins_Heat(:,i) = xmin;
end
mean_time_Heat = toc/N;
percentage_Heat = glob_min/N; 
plot3(mins_Heat(1,:),mins_Heat(2,:),F(mins_Heat),'g*') 




