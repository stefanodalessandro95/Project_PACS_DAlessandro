#include "../Headers/OptimizationFunction.h"
#include "../Headers/MinimizationAlgorithm.h"
#include "../Headers/EntropySGD.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

// Constructors
EntropySGD::EntropySGD(unsigned int dim, OptimizationFunction* F_min, Eigen::VectorXd start, unsigned int max_ep, unsigned int mb, unsigned int L_it,
  double tol_f, double b, double g0, double g1, double y, double x, double a, double r)
  :MinimizationAlgorithm(dim,F_min,start){set_parameters(max_ep,mb,L_it,tol_f,b,g0,g1,y,x,a,r);}

EntropySGD::EntropySGD(unsigned int dim, unsigned int max_ep, unsigned int mb, unsigned int L_it,
  double tol_f, double b, double g0, double g1, double y, double x, double a, double r)
  :MinimizationAlgorithm(dim){set_parameters(max_ep,mb,L_it,tol_f,b,g0,g1,y,x,a,r);}

EntropySGD::EntropySGD(OptimizationFunction* F_min, Eigen::VectorXd start, unsigned int max_ep, unsigned int mb, unsigned int L_it,
  double tol_f, double b, double g0, double g1, double y, double x, double a, double r)
  :MinimizationAlgorithm(F_min,start){set_parameters(max_ep,mb,L_it,tol_f,b,g0,g1,y,x,a,r);}

EntropySGD::EntropySGD(OptimizationFunction* F_min, unsigned int max_ep, unsigned int mb, unsigned int L_it,
  double tol_f, double b, double g0, double g1, double y, double x, double a, double r)
  :MinimizationAlgorithm(F_min){set_parameters(max_ep,mb,L_it,tol_f,b,g0,g1,y,x,a,r);}

EntropySGD::EntropySGD(unsigned int dim, OptimizationFunction* F_min, Eigen::VectorXd start)
  :MinimizationAlgorithm(dim,F_min,start){}

EntropySGD::EntropySGD(unsigned int dim)
  :MinimizationAlgorithm(dim){}

EntropySGD::EntropySGD(OptimizationFunction* F_min, Eigen::VectorXd start)
  :MinimizationAlgorithm(F_min,start){}

EntropySGD::EntropySGD(OptimizationFunction* F_min)
  :MinimizationAlgorithm(F_min){}

/// Set parameters for the algorithm. This re-writes the values that were
/// previously stored.
void EntropySGD::set_parameters(unsigned int max_ep, unsigned int mb, unsigned int L_it,
  double tol_f, double b, double g0, double g1, double y, double x, double a, double r){
    max_epochs = max_ep;
    mb_size = mb;
    L_iterations = L_it;
    tolerance_f = tol_f;
    beta = b;
    gamma0 = g0;
    gamma1 = g1;
    y_step = y;
    x_step = x;
    alpha = a;
    rho = r;
}

/// Method that runs the algorithm. The final value of the parameters that minimizes the
/// function is stored in min_point, which will be accessible with the getter method.
void EntropySGD::solve(){

  if(F_minimization==NULL){
    std::cout << "F not defined" << std::endl;
    return;
  }

  if(starting_point.size()==0){
    std::cout << "Starting point not defined" << std::endl;
    return;
  }

  //std::cout << "Starting Entropy SGD minimization algorithm" << std::endl;
  Function_values_sequence.clear();
  auto start = std::chrono::high_resolution_clock::now();
  Eigen::VectorXd x = starting_point;
  iterations = 1;
  double fprev = F_minimization->evaluate(x)+tolerance_f+1;
  unsigned int N_data = F_minimization->get_data_dim();
  double increment = std::sqrt(y_step/beta);
  Eigen::VectorXd g;

  while(iterations*(L_iterations+1*((double)(rho!=0)))*mb_size/N_data<=max_epochs && std::abs(F_minimization->evaluate(x)-fprev)>tolerance_f){
    double gamma = gamma0*std::pow(1-gamma1,(double)iterations/L_iterations);
    if((iterations*(L_iterations+(unsigned int)(rho!=0))*mb_size)%(3*N_data)==0)
      x_step = x_step/5;

    Eigen::VectorXd y = x;
    Eigen::VectorXd y_mean = x;
    for(unsigned int i=1; i<=L_iterations;i++){
      g = F_minimization->stochastic_gradient(y,mb_size);
      y = y-y_step*(g+(1/gamma)*(y-x))+increment*epsilon();
      y_mean = (1-alpha)*y_mean+alpha*y;
    }

    g = F_minimization->stochastic_gradient(x,mb_size);
    fprev = F_minimization->evaluate(x);
    Function_values_sequence.push_back(fprev);
    x = x-x_step*(g*rho+(x-y_mean)/gamma);
    iterations++;
  }

  min_point = x;
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
  comp_time = duration.count()/1000.0;
  Function_values_sequence.push_back(F_minimization->evaluate(x));
}

/// Returns the number of epochs, so the number of visits to the whole dataset.
double EntropySGD::get_epochs() const {
  if(iterations == 0 || F_minimization == NULL){
    std::cout << "The algorithm has not been performed yet" << std::endl;
    return 0.0;
  }
  return (double) iterations*(L_iterations+1*((double)(rho!=0)))*mb_size/F_minimization->get_data_dim();
}

Eigen::VectorXd EntropySGD::epsilon() const {
  if(state_dim == 0)
    return Eigen::VectorXd::Constant(1,0.0);

  std::random_device rd; //seed
  std::default_random_engine gen(rd()); //engine
  std::normal_distribution<> norm(0,1); //distribution

  Eigen::VectorXd result(state_dim);
  for(unsigned int i=0; i<state_dim; i++)
      result(i)=norm(gen);

  return result;
}
