#ifndef HAVE_GRADIENT_DESCENT_ALGORITHM_H
#define HAVE_GRADIENT_DESCENT_ALGORITHM_H

#include "FunctionData1D.h"
#include <Eigen/Dense>
#include <vector>

/// This class implements the Gradient Descent algorithm for the minimization of functions.
/// This class was not derived from the class MinimizationAlgorithm because the function
/// thet is minimized needs to provide also a method for the evaluation of the gradient. 
class
GradientDescent{

private:
  unsigned int state_dim=1;
  FunctionData1D* F_minimization=NULL;
  Eigen::VectorXd min_point;
  Eigen::VectorXd starting_point;
  unsigned int iterations=0;
  std::vector<double> Function_values_sequence;
  double tolerance_t=0.000001;
  double tolerance_f=0.000001;
  double t0=2.0/3.0;
  double beta=0.8;
  double alpha=0.3;
  double comp_time=0.0;

  Eigen::VectorXd epsilon() const;

public:
  // Constructors
  GradientDescent(unsigned int, FunctionData1D*, Eigen::VectorXd);
  GradientDescent(unsigned int);
  GradientDescent(FunctionData1D*, Eigen::VectorXd);
  GradientDescent(FunctionData1D*);
  GradientDescent(unsigned int, FunctionData1D*, Eigen::VectorXd, double, double, double, double, double);
  GradientDescent(unsigned int, double, double, double, double, double);
  GradientDescent(FunctionData1D*, Eigen::VectorXd, double, double, double, double, double);
  GradientDescent(FunctionData1D*, double, double, double, double, double);

  // Set methods
  void set_state_dim(unsigned int);
  void set_state_dim_no_matter_what(unsigned int);
  void set_function(FunctionData1D*);
  void set_function_no_matter_what(FunctionData1D*);
  void set_starting_point(Eigen::VectorXd);
  void set_parameters(double, double, double, double, double);

  // Get methods
  Eigen::VectorXd get_min() const;
  unsigned int get_iterations() const;
  double get_final_value() const;
  double get_epochs() const;
  double get_computation_time() const;

  // solve method
  void solve();

};

#endif
