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
#include <memory>

double f_1D(Eigen::VectorXd, double);
Eigen::VectorXd df_1D(Eigen::VectorXd, double);

int main(){
  // Dataset
  std::random_device rd; // random seed
  std::default_random_engine re(rd()); // random engine
  std::uniform_real_distribution<double> unif_data(0.0,50.0); //distribution for data
  std::shared_ptr<std::vector<double>> dataset = std::make_shared<std::vector<double>>(*(new(std::vector<double>)));
  unsigned int D = 1000;
  for(unsigned int i=0; i<D/2; i++){
    dataset->push_back(unif_data(re));
    dataset->push_back(unif_data(re)+70.0);
  }

  // Starting point
  std::uniform_real_distribution<double> unif(1.4,2.6); //distribution for starting point
  Eigen::VectorXd xx = Eigen::VectorXd::Constant(1,unif(re));

  FunctionData1D function1D(1,f_1D,df_1D);
  std::cout << "Function 1D" << std::endl;
  function1D.set_dataset(dataset);
  std::cout << std::endl;

  // Gradient Descent
  std::cout << "GRADIENT DESCENT" << std::endl;
  GradientDescent gd_1D(&function1D,0.000001,0.000001,0.4,0.8,0.3);
  int N = 1000; // Number of trials
  int glob_min = 0; // Number of trials converged to the global minimum
  int other_min = 0; // Number of trials converged to other minimum
  double total_comp_time = 0.0;
  for(int i=1; i<=N; i++){
      Eigen::VectorXd xx = Eigen::VectorXd::Constant(1,unif(re));
      gd_1D.set_starting_point(xx);
      gd_1D.solve();
      Eigen::VectorXd x_min = gd_1D.get_min();
      total_comp_time+=gd_1D.get_computation_time();
      if(function1D.evaluate(x_min)<1.02)
        glob_min++;
      else if((x_min(0)>=1.85 && x_min(0)<=1.92)||(x_min(0)>=2.17 && x_min(0)<=2.23))
        other_min++;
  }
  double mean_computation_time_GD_1D = total_comp_time/N;
  double percentage_glob_min_GD_1D = (double)glob_min/N;
  double percentage_other_min_GD_1D = (double)other_min/N;
  std::cout << "Number of trials: " << N << std::endl;
  std::cout << "Mean computation time: " << mean_computation_time_GD_1D << std::endl;
  std::cout << "Percentage of convergence to global minimum: " <<
    percentage_glob_min_GD_1D << std::endl;
  std::cout << "Percentage of convergence to wider minimum: " <<
    percentage_other_min_GD_1D << std::endl << std::endl;

  // Stochastic Gradient Descent
  std::cout << "STOCHASTIC GRADIENT DESCENT" << std::endl;
  StochGradDesc sgd_1D(&function1D,500,100,0.000001,0.00001,0.8,0.15);
  glob_min = 0;
  other_min = 0;
  total_comp_time = 0.0;
  for(int i=1; i<=N; i++){
      Eigen::VectorXd xx = Eigen::VectorXd::Constant(1,unif(re));
      sgd_1D.set_starting_point(xx);
      sgd_1D.solve();
      Eigen::VectorXd x_min = sgd_1D.get_min();
      total_comp_time+=sgd_1D.get_computation_time();
      if(function1D.evaluate(x_min)<0.1)
        glob_min++;
      else if((x_min(0)>=1.85 && x_min(0)<=1.92)||(x_min(0)>=2.17 && x_min(0)<=2.23))
        other_min++;
  }
  double mean_computation_time_SGD_1D = total_comp_time/N;
  double percentage_glob_min_SGD_1D = (double)glob_min/N;
  double percentage_other_min_SGD_1D = (double)other_min/N;
  std::cout << "Number of trials: " << N << std::endl;
  std::cout << "Mean computation time: " << mean_computation_time_SGD_1D << std::endl;
  std::cout << "Percentage of convergence to global minimum: " <<
    percentage_glob_min_SGD_1D << std::endl;
  std::cout << "Percentage of convergence to wider minimum: " <<
    percentage_other_min_SGD_1D << std::endl << std::endl;

  // Entropy SGD
  std::cout << "ENTROPY SGD" << std::endl;
  EntropySGD Esgd_1D(&function1D,200,100,15,0.0001,800,0.2,0.01,0.01,0.005,0.8,0.9);
  glob_min = 0;
  other_min = 0;
  total_comp_time = 0.0;
  for(int i=1; i<=N; i++){
      Eigen::VectorXd xx = Eigen::VectorXd::Constant(1,unif(re));
      Esgd_1D.set_starting_point(xx);
      Esgd_1D.solve();
      Eigen::VectorXd x_min = Esgd_1D.get_min();
      total_comp_time+=Esgd_1D.get_computation_time();
      if(function1D.evaluate(x_min)<0.1)
        glob_min++;
      else if((x_min(0)>= 1.86 && x_min(0)<= 1.90)||(x_min(0)>= 2.17 && x_min(0)<= 2.21))
        other_min++;
  }
  double mean_computation_time_ESGD_1D = total_comp_time/N;
  double percentage_glob_min_ESGD_1D = (double)glob_min/N;
  double percentage_other_min_ESGD_1D = (double)other_min/N;
  std::cout << "Number of trials: " << N << std::endl;
  std::cout << "Mean computation time: " << mean_computation_time_ESGD_1D << std::endl;
  std::cout << "Percentage of convergence to global minimum: " <<
    percentage_glob_min_ESGD_1D << std::endl;
  std::cout << "Percentage of convergence to wider minimum: " <<
    percentage_other_min_ESGD_1D << std::endl << std::endl;

  // Heat
  std::cout << "HEAT SGD" << std::endl;
  Heat Heat_1D(&function1D,200,100,20,0.0001,0.0001,0.00001,0.01);
  glob_min = 0;
  other_min = 0;
  total_comp_time = 0.0;
  for(int i=1; i<=N; i++){
      Eigen::VectorXd xx = Eigen::VectorXd::Constant(1,unif(re));
      Heat_1D.set_starting_point(xx);
      Heat_1D.solve();
      Eigen::VectorXd x_min = Heat_1D.get_min();
      total_comp_time+=Heat_1D.get_computation_time();
      if(function1D.evaluate(x_min)<0.1)
        glob_min++;
      else if((x_min(0)>=1.86 && x_min(0)<=1.90)||(x_min(0)>=2.17 && x_min(0)<=2.21))
        other_min++;
  }
  double mean_computation_time_HEAT_1D = total_comp_time/N;
  double percentage_glob_min_HEAT_1D = (double)glob_min/N;
  double percentage_other_min_HEAT_1D = (double)other_min/N;
  std::cout << "Number of trials: " << N << std::endl;
  std::cout << "Mean computation time: " << mean_computation_time_HEAT_1D << std::endl;
  std::cout << "Percentage of convergence to global minimum: " <<
    percentage_glob_min_HEAT_1D << std::endl;
  std::cout << "Percentage of convergence to wider minimum: " <<
    percentage_other_min_HEAT_1D << std::endl << std::endl;

  return 0;
}

double f_1D(Eigen::VectorXd x, double y){
  if(x.size()>=1){
    double xx = x(0);
    double g = std::sin(20.0)*std::cos(8.0*y)-std::sin(6.0*y)+0.02*std::abs(y);
    double f = (std::sin(10.0*xx)*std::cos(4.0*xx*y)-std::sin(6.0*y))*std::exp(-5.0*(xx-2.0)*(xx-2.0));
    return 0.5*(g-f)*(g-f);
  } else
    return 0;
}

Eigen::VectorXd df_1D(Eigen::VectorXd x, double y){
  if(x.size()>=1){
    double xx = x(0);
    double g = std::sin(20.0)*std::cos(8.0*y)-std::sin(6.0*y)+0.02*std::abs(y);
    double f = (std::sin(10.0*xx)*std::cos(4.0*xx*y)-std::sin(6.0*y))*std::exp(-5.0*(xx-2.0)*(xx-2.0));
    double df = (10.0*std::cos(10.0*xx)*std::cos(4.0*xx*y)-4.0*y*std::sin(10.0*xx)*std::sin(4.0*xx*y))*std::exp(-5.0*(xx-2.0)*(xx-2.0))-10*(xx-2.0)*f;
    return -(g-f)*Eigen::VectorXd::Constant(1,df);
  } else
    return Eigen::VectorXd::Zero(1);
}
