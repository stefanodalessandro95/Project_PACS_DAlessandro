#include "../Headers/OptimizationFunction.h"
#include "../Headers/MinimizationAlgorithm.h"
#include "../Headers/Heat.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

// Constructors
Heat::Heat(unsigned int dim, OptimizationFunction* F_min, Eigen::VectorXd start, unsigned int max_ep, unsigned int mb, unsigned int L_it,
  double tol_f, double g0, double g1, double x)
  :MinimizationAlgorithm(dim,F_min,start){set_parameters(max_ep,mb,L_it,tol_f,g0,g1,x);}

Heat::Heat(unsigned int dim, unsigned int max_ep, unsigned int mb, unsigned int L_it,
  double tol_f, double g0, double g1, double x)
  :MinimizationAlgorithm(dim){set_parameters(max_ep,mb,L_it,tol_f,g0,g1,x);}

Heat::Heat(OptimizationFunction* F_min, Eigen::VectorXd start, unsigned int max_ep, unsigned int mb, unsigned int L_it,
  double tol_f, double g0, double g1, double x)
  :MinimizationAlgorithm(F_min,start){set_parameters(max_ep,mb,L_it,tol_f,g0,g1,x);}

Heat::Heat(OptimizationFunction* F_min, unsigned int max_ep, unsigned int mb, unsigned int L_it,
  double tol_f, double g0, double g1, double x)
  :MinimizationAlgorithm(F_min){set_parameters(max_ep,mb,L_it,tol_f,g0,g1,x);}

Heat::Heat(unsigned int dim, OptimizationFunction* F_min, Eigen::VectorXd start)
  :MinimizationAlgorithm(dim,F_min,start){}

Heat::Heat(unsigned int dim)
  :MinimizationAlgorithm(dim){}

Heat::Heat(OptimizationFunction* F_min, Eigen::VectorXd start)
  :MinimizationAlgorithm(F_min,start){}

Heat::Heat(OptimizationFunction* F_min)
  :MinimizationAlgorithm(F_min){}

/// Set parameters for the algorithm. This re-writes the values that were
/// previously stored.
void Heat::set_parameters(unsigned int max_ep, unsigned int mb, unsigned int L_it,
  double tol_f, double g0, double g1, double x){
    max_epochs = max_ep;
    mb_size = mb;
    L_iterations = L_it;
    tolerance_f = tol_f;
    gamma0 = g0;
    gamma1 = g1;
    x_step_0 = x;
}

/// Method that runs the algorithm. The final value of the parameters that minimizes the
/// function is stored in min_point, which will be accessible with the getter method.
void Heat::solve(){

  if(F_minimization==NULL){
    std::cout << "F not defined" << std::endl;
    return;
  }

  if(starting_point.size()==0){
    std::cout << "Starting point not defined" << std::endl;
    return;
  }

  //std::cout << "Starting Heat minimization algorithm" << std::endl;
  Function_values_sequence.clear();
  auto start = std::chrono::high_resolution_clock::now();
  Eigen::VectorXd x = starting_point;
  iterations = 1;
  double fprev = F_minimization->evaluate(x)+tolerance_f+1;
  unsigned int N_data = F_minimization->get_data_dim();
  Eigen::VectorXd g = Eigen::VectorXd::Zero(state_dim);
  double x_step = 0.0;
  double gamma = 0.0;

  while(iterations*L_iterations*mb_size/N_data<=max_epochs && std::abs(F_minimization->evaluate(x)-fprev)>tolerance_f){
    gamma = gamma0*(1-std::pow(gamma1,iterations/L_iterations));
    x_step = x_step_0/std::sqrt(iterations*L_iterations*mb_size/N_data);

    for(unsigned int i=1; i<=L_iterations;i++)
      g+= F_minimization->stochastic_gradient(x+gamma*epsilon(),mb_size);
    g = g/L_iterations;

    fprev = F_minimization->evaluate(x);
    Function_values_sequence.push_back(fprev);
    x = x-x_step*g;
    iterations++;
  }

  min_point = x;
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
  comp_time = duration.count()/1000.0;
  Function_values_sequence.push_back(F_minimization->evaluate(x));
}

/// Returns the number of epochs, so the number of visits to the whole dataset.
double Heat::get_epochs() const {
  if(iterations == 0 || F_minimization == NULL){
    std::cout << "The algorithm has not been performed yet" << std::endl;
    return 0.0;
  }
  return (double) iterations*L_iterations*mb_size/F_minimization->get_data_dim();
}

Eigen::VectorXd Heat::epsilon() const {
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
