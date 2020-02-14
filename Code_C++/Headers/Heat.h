#ifndef HAVE_HEAT_ALGORITHM_H
#define HAVE_HEAT_ALGORITHM_H

#include "OptimizationFunction.h"
#include "MinimizationAlgorithm.h"
#include <Eigen/Dense>

/// This class is derived from the abstract class MinimizationAlgorithm.
/// This implements the Heat SGD algorithm.
class
Heat: public MinimizationAlgorithm{

private:
  unsigned int max_epochs=200;
  unsigned int mb_size=100;
  unsigned int L_iterations=20;
  double tolerance_f=0.00001;
  double gamma0=0.0001;
  double gamma1=0.00001;
  double x_step_0=0.01;

  Eigen::VectorXd epsilon() const;

public:
  // Constructors
  Heat(unsigned int, OptimizationFunction*, Eigen::VectorXd, unsigned int, unsigned int, unsigned int,
    double, double, double, double);
  Heat(unsigned int, unsigned int, unsigned int, unsigned int,
    double, double, double, double);
  Heat(OptimizationFunction*, Eigen::VectorXd, unsigned int, unsigned int, unsigned int,
    double, double, double, double);
  Heat(OptimizationFunction*, unsigned int, unsigned int, unsigned int,
    double, double, double, double);
  Heat(unsigned int, OptimizationFunction*, Eigen::VectorXd);
  Heat(unsigned int);
  Heat(OptimizationFunction*, Eigen::VectorXd);
  Heat(OptimizationFunction*);

  // set methods
  void set_parameters(unsigned int, unsigned int, unsigned int, double, double, double, double);

  // Overridden methods from abstract base class
  void solve() override;

  // get methods
  double get_epochs() const override;

};

#endif
