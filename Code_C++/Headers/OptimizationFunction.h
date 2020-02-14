#ifndef HAVE_OPTIMIZATION_FUNCTION_H
#define HAVE_OPTIMIZATION_FUNCTION_H

#include <Eigen/Dense>

/// This is an abstract class that is used as a model for functions to be minimized
/// by minimization algorithms. In particular the classes derived from this class
/// must provide a stochastic_gradient method for the evaluation of the stochastic
/// gradient, given a state (a set of parameters) and a dimention for the mini batch.
class
OptimizationFunction
{

private :
  unsigned int state_dim=1;

public :
  // Constructors
  OptimizationFunction()=default;
  explicit OptimizationFunction(unsigned int dim):state_dim(dim){}

  // Get method
  /// Returns the dimention of the state, or the number of the parameters for the
  /// optimization.
  unsigned int get_state_dim() const {return state_dim;}

  // Set method
  /// Re-writes the previous value of the state_dim;
  void set_state_dim(unsigned int dim){state_dim=dim;}

  // Pure virtual methods -> abstract class
  // These methods are ovverridden by the derived classes, according to their
  // internal structure. 
  virtual double evaluate(const Eigen::VectorXd &) =0;
  virtual Eigen::VectorXd stochastic_gradient(const Eigen::VectorXd &,unsigned int) =0;
  virtual unsigned int get_data_dim() const =0;
};


#endif
