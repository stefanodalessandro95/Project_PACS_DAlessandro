#include "../Headers/MinimizationAlgorithm.h"
#include "../Headers/OptimizationFunction.h"
#include <Eigen/Dense>
#include <iostream>

// Constructors
MinimizationAlgorithm::MinimizationAlgorithm(unsigned int dim, OptimizationFunction* F_min, Eigen::VectorXd start)
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

MinimizationAlgorithm::MinimizationAlgorithm(unsigned int dim):state_dim(dim){}

MinimizationAlgorithm::MinimizationAlgorithm(OptimizationFunction* F_min, Eigen::VectorXd start)
  :F_minimization(F_min){
    if(F_minimization != NULL)
      state_dim = F_minimization->get_state_dim();
    if(start.size() == state_dim){
      starting_point = start;
    } else {
      std::cout << "Starting point dimention not coherent" << std::endl;
    }
  }

MinimizationAlgorithm::MinimizationAlgorithm(OptimizationFunction* F_min):F_minimization(F_min){
  if(F_minimization != NULL)
    state_dim = F_minimization->get_state_dim();
}

// Set methods
/// This method sets the dimention of the state, but only if the function to be
/// optimized is not already set.
void MinimizationAlgorithm::set_state_dim(unsigned int dim){
  if(F_minimization == NULL)
    state_dim = dim;
}

/// This algorithm changes the dimention of the state, even if the new dimention
/// is not coherent with the dimention of the OptimizationFunction. In that case
/// the pointer to the OptimizationFunction is set as NULL.
void MinimizationAlgorithm::set_state_dim_no_matter_what(unsigned int dim){
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
void MinimizationAlgorithm::set_function(OptimizationFunction* F_min){
  if(F_min->get_state_dim() == state_dim)
    F_minimization = F_min;
}

/// This method changes the pointer to the OptimizationFunction, even if the
/// dimention of the state and the dimention of the new function are not coherent.
/// In this case the new dimention is set to the dimention of the state of the
/// new function.
void MinimizationAlgorithm::set_function_no_matter_what(OptimizationFunction* F_min){
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
void MinimizationAlgorithm::set_starting_point(Eigen::VectorXd start){
  if(start.size() == state_dim){
    starting_point = start;
  } else {
    std::cout << "Starting point dimention not coherent" << std::endl;
  }
}

// Get methods
/// Returns the final value of the parameters, found and the end of the algorithm.
Eigen::VectorXd MinimizationAlgorithm::get_min() const {
  if(min_point.size()==0)
    std::cout << "The algorithm has not been performed yet" << std::endl;
  return min_point;
}

/// Returns the final value of the function found at the end of the algorithm.
/// This value of the function corresponds to the value computed at min_point.
double MinimizationAlgorithm::get_final_value() const {
  if(iterations == 0){
    std::cout << "The algorithm has not been performed yet" << std::endl;
    return 0.0;
  }
  return Function_values_sequence.back();
}

/// Returns the iterations needed for the minimization algorithm.
unsigned int MinimizationAlgorithm::get_iterations() const {
  if(iterations == 0){
    std::cout << "The algorithm has not been performed yet" << std::endl;
  }
  return iterations;
}

/// Returns the computation time for the minimization, in milliseconds.
double MinimizationAlgorithm::get_computation_time() const {
  if(comp_time == 0.0){
    std::cout << "The algorithm has not been performed yet" << std::endl;
  }
  return comp_time;
}
