#include "../Headers/FunctionData1D.h"
#include "../Headers/GradientDescent.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <chrono>

// Constructors
GradientDescent::GradientDescent(unsigned int dim, FunctionData1D* F_min, Eigen::VectorXd start)
  :state_dim(dim){
    if(F_min->get_state_dim() == state_dim){
      F_minimization = F_min;
    } else {
      std::cout << "Function dimention not coherent" << std::endl;
    }

    if(start.size() == state_dim){
      starting_point = start;
    } else {
      std::cout << "Starting point dimention not coherent" << std::endl;
    }
  }

GradientDescent::GradientDescent(unsigned int dim):state_dim(dim){}

GradientDescent::GradientDescent(FunctionData1D* F_min, Eigen::VectorXd start)
  :F_minimization(F_min){
    if(F_minimization != NULL)
      state_dim = F_minimization->get_state_dim();
    if(start.size() == state_dim){
      starting_point = start;
    } else {
      std::cout << "Starting point dimention not coherent" << std::endl;
    }
  }

GradientDescent::GradientDescent(FunctionData1D* F_min):F_minimization(F_min){
  if(F_minimization != NULL)
    state_dim = F_minimization->get_state_dim();
}

GradientDescent::GradientDescent(unsigned int dim, FunctionData1D* F_min, Eigen::VectorXd start,
  double tol_t, double tol_f, double t, double b, double a)
  :GradientDescent(dim,F_min,start){set_parameters(tol_t, tol_f, t, b, a);}

GradientDescent::GradientDescent(unsigned int dim,
  double tol_t, double tol_f, double t, double b, double a)
  :GradientDescent(dim){set_parameters(tol_t, tol_f, t, b, a);}

GradientDescent::GradientDescent(FunctionData1D* F_min, Eigen::VectorXd start,
  double tol_t, double tol_f, double t, double b, double a)
  :GradientDescent(F_min,start){set_parameters(tol_t, tol_f, t, b, a);}

GradientDescent::GradientDescent(FunctionData1D* F_min,
  double tol_t, double tol_f, double t, double b, double a)
  :GradientDescent(F_min){set_parameters(tol_t, tol_f, t, b, a);}

// Set methods
/// This method sets the dimention of the state, but only if the function to be
/// optimized is not already set.
void GradientDescent::set_state_dim(unsigned int dim){
  if(F_minimization == NULL)
    state_dim = dim;
}

/// This algorithm changes the dimention of the state, even if the new dimention
/// is not coherent with the dimention of the FunctionData1D. In that case
/// the pointer to the FunctionData1D is set as NULL.
void GradientDescent::set_state_dim_no_matter_what(unsigned int dim){
    state_dim = dim;
    if(F_minimization->get_state_dim() != state_dim)
      F_minimization = NULL;
    if(starting_point.size() != state_dim && starting_point.size() != 0){
      Eigen::VectorXd temp;
      starting_point = temp;
    }
    if(min_point.size() != state_dim && min_point.size() != 0){
      Eigen::VectorXd temp;
      min_point = temp;
    }
}

/// This method changes the pointer to the function to be optimized, but only if
/// the dimention of the new function is equal to the dimention of the state.
void GradientDescent::set_function(FunctionData1D* F_min){
  if(F_min->get_state_dim() == state_dim)
    F_minimization = F_min;
}

/// This method changes the pointer to the FunctionData1D, even if the
/// dimention of the state and the dimention of the new function are not coherent.
/// In this case the new dimention is set to the dimention of the state of the
/// new function.
void GradientDescent::set_function_no_matter_what(FunctionData1D* F_min){
  F_minimization = F_min;
  if(F_minimization == NULL){
    state_dim = 1;
    Eigen::VectorXd temp;
    starting_point = temp;
    return;
  }
  if(F_minimization->get_state_dim() != state_dim)
    state_dim = F_minimization->get_state_dim();
  if(starting_point.size() != F_minimization->get_state_dim() && starting_point.size() != 0){
    Eigen::VectorXd temp;
    starting_point = temp;
  }
  if(min_point.size() != F_minimization->get_state_dim() && min_point.size() != 0){
    Eigen::VectorXd temp;
    min_point = temp;
  }
}

/// Method to set the starting point of the algorithm. The dimention of the starting point
/// must match with the dimention of the state.
void GradientDescent::set_starting_point(Eigen::VectorXd start){
  if(start.size() == state_dim){
    starting_point = start;
  } else {
    std::cout << "Starting point dimention not coherent" << std::endl;
  }
}

/// Set parameters for the algorithm. This re-writes the values that were
/// previously stored.
void GradientDescent::set_parameters(double tol_t, double tol_f, double t, double b, double a){
    tolerance_t = tol_t;
    tolerance_f = tol_f;
    t0 = t;
    beta = b;
    alpha = a;
}

// Solve Method
/// Method that runs the algorithm. The final value of the parameters that minimizes the
/// function is stored in min_point, which will be accessible with the getter method.
void GradientDescent::solve(){

  if(F_minimization==NULL){
    std::cout << "F not defined" << std::endl;
    return;
  }

  if(starting_point.size()==0){
    std::cout << "Starting point not defined" << std::endl;
    return;
  }

  //std::cout << "Starting Gradient Descent minimization algorithm" << std::endl;
  Function_values_sequence.clear();
  auto start = std::chrono::high_resolution_clock::now();
  Eigen::VectorXd x = starting_point;
  iterations = 1;
  double fprev = F_minimization->evaluate(x)+tolerance_f+1.0;
  double t = t0;
  Eigen::VectorXd g;
  Eigen::VectorXd g_dir;

  while(t>tolerance_t && std::abs(F_minimization->evaluate(x)-fprev)>tolerance_f){
    g = F_minimization->gradient(x);
    g_dir = -g/g.norm();
    t = t0;
    fprev = F_minimization->evaluate(x);

    while(F_minimization->evaluate(x+t*g_dir) > fprev + alpha*t*g.dot(g_dir))
      t = beta*t;

    Function_values_sequence.push_back(fprev);
    x+=t*g_dir;
    iterations++;
  }

  min_point = x;
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
  comp_time = duration.count()/1000.0;
  Function_values_sequence.push_back(F_minimization->evaluate(x));
}

// Get methods
/// Returns the final value of the parameters, found and the end of the algorithm.
Eigen::VectorXd GradientDescent::get_min() const {
  if(min_point.size()==0)
    std::cout << "The algorithm has not been performed yet" << std::endl;
  return min_point;
}

/// Returns the final value of the function found at the end of the algorithm.
/// This value of the function corresponds to the value computed at min_point.
double GradientDescent::get_final_value() const {
  if(iterations == 0){
    std::cout << "The algorithm has not been performed yet" << std::endl;
    return 0.0;
  }
  return Function_values_sequence.back();
}

/// Returns the iterations needed for the minimization algorithm.
unsigned int GradientDescent::get_iterations() const {
  if(iterations == 0){
    std::cout << "The algorithm has not been performed yet" << std::endl;
  }
  return iterations;
}

/// Returns the number of epochs, so the number of visits to the whole dataset.
double GradientDescent::get_epochs() const {
  if(iterations == 0 || F_minimization == NULL){
    std::cout << "The algorithm has not been performed yet" << std::endl;
    return 0.0;
  }
  return (double) iterations;
}

/// Returns the computation time for the minimization, in milliseconds.
double GradientDescent::get_computation_time() const {
  if(comp_time == 0.0){
    std::cout << "The algorithm has not been performed yet" << std::endl;
  }
  return comp_time;
}
