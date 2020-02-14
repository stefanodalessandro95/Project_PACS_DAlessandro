#include "../Headers/OptimizationFunction.h"
#include "../Headers/MinimizationAlgorithm.h"
#include "../Headers/StochGradDesc.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <chrono>

// Constructors
StochGradDesc::StochGradDesc(unsigned int dim, OptimizationFunction* F_min, Eigen::VectorXd start,
  unsigned int max_it, unsigned int mb, double tol_t, double tol_f, double b, double t)
  :MinimizationAlgorithm(dim,F_min,start){set_parameters(max_it,mb,tol_t,tol_f,b,t);}

StochGradDesc::StochGradDesc(unsigned int dim,
  unsigned int max_it, unsigned int mb, double tol_t, double tol_f, double b, double t)
  :MinimizationAlgorithm(dim){set_parameters(max_it,mb,tol_t,tol_f,b,t);}

StochGradDesc::StochGradDesc(OptimizationFunction* F_min, Eigen::VectorXd start,
  unsigned int max_it, unsigned int mb, double tol_t, double tol_f, double b, double t)
  :MinimizationAlgorithm(F_min,start){set_parameters(max_it,mb,tol_t,tol_f,b,t);}

StochGradDesc::StochGradDesc(OptimizationFunction* F_min,
  unsigned int max_it, unsigned int mb, double tol_t, double tol_f, double b, double t)
  :MinimizationAlgorithm(F_min){set_parameters(max_it,mb,tol_t,tol_f,b,t);}

StochGradDesc::StochGradDesc(unsigned int dim, OptimizationFunction* F_min, Eigen::VectorXd start)
  :MinimizationAlgorithm(dim,F_min,start){}

StochGradDesc::StochGradDesc(unsigned int dim)
  :MinimizationAlgorithm(dim){}

StochGradDesc::StochGradDesc(OptimizationFunction* F_min, Eigen::VectorXd start)
  :MinimizationAlgorithm(F_min,start){}

StochGradDesc::StochGradDesc(OptimizationFunction* F_min)
  :MinimizationAlgorithm(F_min){}

/// Set parameters for the algorithm. This re-writes the values that were
/// previously stored.
void StochGradDesc::set_parameters(unsigned int max_it, unsigned int mb, double tol_t, double tol_f, double b, double t){
    max_iterations = max_it;
    mb_size = mb;
    tolerance_t = tol_t;
    tolerance_f = tol_f;
    beta = b;
    t_zero = t;
}

/// Method that runs the algorithm. The final value of the parameters that minimizes the
/// function is stored in min_point, which will be accessible with the getter method.
void StochGradDesc::solve(){
  if(F_minimization==NULL){
    std::cout << "F not defined" << std::endl;
    return;
  }

  if(starting_point.size()==0){
    std::cout << "Starting point not defined" << std::endl;
    return;
  }

  //std::cout << "Starting SGD minimization algorithm" << std::endl;
  Function_values_sequence.clear();
  auto start = std::chrono::high_resolution_clock::now();
  Eigen::VectorXd x = starting_point;
  iterations = 1;
  double t = t_zero;
  double fprev = F_minimization->evaluate(x)+tolerance_f+1;
  unsigned int N_data = F_minimization->get_data_dim();
  Eigen::VectorXd g;

  while(t>tolerance_t && std::abs(F_minimization->evaluate(x)-fprev)>tolerance_f && iterations<max_iterations){

      t = std::pow(t_zero,(double) 1.0+iterations*mb_size/N_data);
      g = -t*(F_minimization->stochastic_gradient(x,mb_size));
      fprev = F_minimization->evaluate(x);

      while((F_minimization->evaluate(x+g)>=fprev || g.norm()>2) && g.norm()>tolerance_t){
        t = beta*t;
        g = -t*(F_minimization->stochastic_gradient(x,mb_size));
        if(t<tolerance_t)
          g = Eigen::VectorXd::Zero(state_dim);
      }

      Function_values_sequence.push_back(fprev);
      x+=g;
      iterations++;
  }

  min_point = x;
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
  comp_time = duration.count()/1000.0;
  Function_values_sequence.push_back(F_minimization->evaluate(x));
}

/// Returns the number of epochs, so the number of visits to the whole dataset.
double StochGradDesc::get_epochs() const {
  if(iterations == 0 || F_minimization == NULL){
    std::cout << "The algorithm has not been performed yet" << std::endl;
    return 0.0;
  }
  return (double) iterations*mb_size/F_minimization->get_data_dim();
}
