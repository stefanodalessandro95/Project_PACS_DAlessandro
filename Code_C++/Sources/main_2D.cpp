#include <stdlib.h>
#include <iostream>
#include "../Headers/OptimizationFunction.h"
#include "../Headers/FunctionData1D.h"
#include "../Headers/MinimizationAlgorithm.h"
#include "../Headers/StochGradDesc.h"
#include "../Headers/EntropySGD.h"
#include "../Headers/Heat.h"
#include "../Headers/GradientDescent.h"
#include <Eigen/Dense>
#include <cmath>
#include <random>

double f_2D(Eigen::VectorXd, double);
Eigen::VectorXd df_2D(Eigen::VectorXd, double);

int main(){
  // 2D functions
  std::random_device rd; //seed
  std::default_random_engine re(rd()); //engine
  std::uniform_real_distribution<double> unif(1,3); //distribution

  Eigen::VectorXd xx(2);
  xx << unif(re), unif(re);
  FunctionData1D function2D(2,f_2D,df_2D);
  std::cout << "Function 2D" << std::endl;
  function2D.make_dataset(1000,0.0,20.0);
  std::cout << std::endl;

  // Gradient Descent
  std::cout << "GRADIENT DESCENT" << std::endl;
  GradientDescent gd_2D(&function2D,0.000001,0.000001,2.0/3.0,0.8,0.3);
  int N = 1000;
  int glob_min = 0;
  double total_comp_time = 0.0;
  for(int i=1; i<=N; i++){
      xx(0) = unif(re);
      xx(1) = unif(re);
      gd_2D.set_starting_point(xx);
      gd_2D.solve();
      Eigen::VectorXd x_min = gd_2D.get_min();
      total_comp_time+=gd_2D.get_computation_time();
      if(function2D.evaluate(x_min)<0.01)
        glob_min++;
  }
  double mean_computation_time_GD_2D = total_comp_time/N;
  double percentage_glob_min_GD_2D = (double)glob_min/N;
  std::cout << "Number of trials: " << N << std::endl;
  std::cout << "Mean computation time: " << mean_computation_time_GD_2D << std::endl;
  std::cout << "Percentage of convergence to global minimum: " <<
    percentage_glob_min_GD_2D << std::endl << std::endl;

  // Stochastic Gradient Descent
  std::cout << "STOCHASTIC GRADIENT DESCENT" << std::endl;
  StochGradDesc sgd_2D(&function2D,500,100,0.000001,0.00001,0.8,0.25);
  glob_min = 0;
  total_comp_time = 0.0;
  for(int i=1; i<=N; i++){
      xx(0) = unif(re);
      xx(1) = unif(re);
      sgd_2D.set_starting_point(xx);
      sgd_2D.solve();
      Eigen::VectorXd x_min = sgd_2D.get_min();
      total_comp_time+=sgd_2D.get_computation_time();
      if(function2D.evaluate(x_min)<0.01)
        glob_min++;
  }
  double mean_computation_time_SGD_2D = total_comp_time/N;
  double percentage_glob_min_SGD_2D = (double)glob_min/N;
  std::cout << "Number of trials: " << N << std::endl;
  std::cout << "Mean computation time: " << mean_computation_time_SGD_2D << std::endl;
  std::cout << "Percentage of convergence to global minimum: " <<
    percentage_glob_min_SGD_2D << std::endl << std::endl;

  // Entropy SGD
  std::cout << "ENTROPY SGD" << std::endl;
  EntropySGD Esgd_2D(&function2D,200,100,10,0.0001,100.0,1,0.1,0.04,0.1,0.9,0.01);
  glob_min = 0;
  total_comp_time = 0.0;
  for(int i=1; i<=N; i++){
      xx(0) = unif(re);
      xx(1) = unif(re);
      Esgd_2D.set_starting_point(xx);
      Esgd_2D.solve();
      Eigen::VectorXd x_min = Esgd_2D.get_min();
      total_comp_time+=Esgd_2D.get_computation_time();
      if(function2D.evaluate(x_min)<0.01)
        glob_min++;
  }
  double mean_computation_time_ESGD_2D = total_comp_time/N;
  double percentage_glob_min_ESGD_2D = (double)glob_min/N;
  std::cout << "Number of trials: " << N << std::endl;
  std::cout << "Mean computation time: " << mean_computation_time_ESGD_2D << std::endl;
  std::cout << "Percentage of convergence to global minimum: " <<
    percentage_glob_min_ESGD_2D << std::endl << std::endl;

  // Heat
  std::cout << "HEAT SGD" << std::endl;
  Heat Heat_2D(&function2D,200,100,20,0.0001,0.05,0.001,0.08);
  glob_min = 0;
  total_comp_time = 0.0;
  for(int i=1; i<=N; i++){
      xx(0) = unif(re);
      xx(1) = unif(re);
      Heat_2D.set_starting_point(xx);
      Heat_2D.solve();
      Eigen::VectorXd x_min = Heat_2D.get_min();
      total_comp_time+=Heat_2D.get_computation_time();
      if(function2D.evaluate(x_min)<0.01)
        glob_min++;
  }
  double mean_computation_time_HEAT_2D = total_comp_time/N;
  double percentage_glob_min_HEAT_2D = (double)glob_min/N;
  std::cout << "Number of trials: " << N << std::endl;
  std::cout << "Mean computation time: " << mean_computation_time_HEAT_2D << std::endl;
  std::cout << "Percentage of convergence to global minimum: " <<
    percentage_glob_min_HEAT_2D << std::endl << std::endl;

  return 0;
}

double f_2D(Eigen::VectorXd x, double y){
  if(x.size()>=2){
    double x1 = x(0);
    double x2 = x(1);
    double g = (std::sin(2.0*std::cos(2.0*y))+std::cos(2.0*std::sin(2.0*y))-std::sin(4.0));
    double f = (std::sin(x1*std::cos(x1*y))+std::cos(x2*std::sin(x2*y))-std::sin(x1*x2))*std::exp(0.5*(x1-2.0)*(x1-2.0)+0.5*(x2-2.0)*(x2-2.0));
    return 0.5*(g-f)*(g-f);
  } else
    return 0;
}

Eigen::VectorXd df_2D(Eigen::VectorXd x, double y){
  if(x.size()>=2){
    double x1 = x(0);
    double x2 = x(1);
    double g = (std::sin(2.0*std::cos(2.0*y))+std::cos(2.0*std::sin(2.0*y)-std::sin(4.0)));
    double f = (std::sin(x1*std::cos(x1*y))+std::cos(x2*std::sin(x2*y)-std::sin(x1*x2)))*std::exp(0.5*(x1-2.0)*(x1-2.0)+0.5*(x2-2.0)*(x2-2.0));
    double df1 = (std::cos(x1*std::cos(x1*y))*(std::cos(x1*y)-y*x1*std::sin(x1*y))+std::sin(x2*std::sin(x2*y)-std::sin(x1*x2))*std::cos(x1*x2)*x2)*std::exp(0.5*(x1-2.0)*(x1-2.0)+0.5*(x2-2.0)*(x2-2.0))+(x1-2.0)*f;
    double df2 = -std::sin(x2*std::sin(x2*y)-std::sin(x1*x2))*(std::sin(x2*y)+x2*y*std::cos(x2*y)-x1*std::cos(x1*x2))*std::exp(0.5*(x1-2.0)*(x1-2.0)+0.5*(x2-2.0)*(x2-2.0))+(x2-2.0)*f;
    Eigen::VectorXd df_vect(2);
    df_vect << df1, df2;
    return -(g-f)*df_vect;
  } else
    return Eigen::VectorXd::Zero(2);
}
