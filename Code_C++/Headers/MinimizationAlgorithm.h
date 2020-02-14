#ifndef HAVE_MINIMIZATION_ALGORITHM_H
#define HAVE_MINIMIZATION_ALGORITHM_H

#include "OptimizationFunction.h"
#include <Eigen/Dense>
#include <vector>

/// This is an abstract class, used as a basis for the development of
/// algorithms for the minimization of functions that depend on a dataset.
/// In particular this class contains a pointer to the abstract class
/// OptimizationFunction. Any class derived from that would possibly be minimized by
/// an algorith, derived from this class. 
class
MinimizationAlgorithm{

protected :
  unsigned int state_dim=1;
  OptimizationFunction* F_minimization=NULL;
  Eigen::VectorXd min_point;
  Eigen::VectorXd starting_point;
  unsigned int iterations=0;
  std::vector<double> Function_values_sequence;
  double comp_time=0.0;

public :
  // Constructors
  MinimizationAlgorithm(unsigned int, OptimizationFunction*, Eigen::VectorXd);
  MinimizationAlgorithm(unsigned int);
  MinimizationAlgorithm(OptimizationFunction*, Eigen::VectorXd);
  MinimizationAlgorithm(OptimizationFunction*);

  // set methods
  void set_state_dim(unsigned int);
  void set_state_dim_no_matter_what(unsigned int);
  void set_function(OptimizationFunction*);
  void set_function_no_matter_what(OptimizationFunction*);
  void set_starting_point(Eigen::VectorXd);

  // get methods
  Eigen::VectorXd get_min() const;
  unsigned int get_iterations() const;
  double get_final_value() const;
  double get_computation_time() const;

  // Pure virtual method -> abstract class
  virtual void solve()=0;
  virtual double get_epochs() const =0;

};

#endif
