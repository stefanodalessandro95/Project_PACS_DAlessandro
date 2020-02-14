#ifndef HAVE_SGD_ALGORITHM_H
#define HAVE_SGD_ALGORITHM_H

#include "OptimizationFunction.h"
#include "MinimizationAlgorithm.h"
#include <Eigen/Dense>

/// This class is derived from the abstract class MinimizationAlgorithm.
/// This implements the Stochastic Gradient Descent algorithm.
class
StochGradDesc: public MinimizationAlgorithm{

private:
  unsigned int max_iterations=1000;
  unsigned int mb_size=100;
  double tolerance_t=0.000001;
  double tolerance_f=0.000001;
  double beta=0.8;
  double t_zero=0.25;

public:
  // Constructors
  StochGradDesc(unsigned int, OptimizationFunction*, Eigen::VectorXd, unsigned int, unsigned int, double, double, double, double);
  StochGradDesc(unsigned int, unsigned int, unsigned int, double, double, double, double);
  StochGradDesc(OptimizationFunction*, Eigen::VectorXd, unsigned int, unsigned int, double, double, double, double);
  StochGradDesc(OptimizationFunction*, unsigned int, unsigned int, double, double, double, double);
  StochGradDesc(unsigned int, OptimizationFunction*, Eigen::VectorXd);
  StochGradDesc(unsigned int);
  StochGradDesc(OptimizationFunction*, Eigen::VectorXd);
  StochGradDesc(OptimizationFunction*);

  // set methods
  void set_parameters(unsigned int, unsigned int, double, double, double, double);

  // Overridden methods from abstract base class
  void solve() override;

  // get methods
  double get_epochs() const override;

};

#endif
