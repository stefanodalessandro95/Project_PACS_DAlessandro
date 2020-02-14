#ifndef HAVE_ENTROPY_SGD_ALGORITHM_H
#define HAVE_ENTROPY_SGD_ALGORITHM_H

#include "OptimizationFunction.h"
#include "MinimizationAlgorithm.h"
#include <Eigen/Dense>

/// This class is derived from the abstract class MinimizationAlgorithm.
/// This implements the Entropy SGD algorithm.
class
EntropySGD: public MinimizationAlgorithm{

private:
  unsigned int max_epochs=200;
  unsigned int mb_size=100;
  unsigned int L_iterations=15;
  double tolerance_f=0.00001;
  double beta=800.0;
  double gamma0=0.2;
  double gamma1=0.01;
  double y_step=0.01;
  double x_step=0.005;
  double alpha=0.8;
  double rho=0.9;

  Eigen::VectorXd epsilon() const;

public:
  // Constructors
  EntropySGD(unsigned int, OptimizationFunction*, Eigen::VectorXd, unsigned int, unsigned int, unsigned int,
    double, double, double, double, double, double, double, double);
  EntropySGD(unsigned int, unsigned int, unsigned int, unsigned int,
    double, double, double, double, double, double, double, double);
  EntropySGD(OptimizationFunction*, Eigen::VectorXd, unsigned int, unsigned int, unsigned int,
    double, double, double, double, double, double, double, double);
  EntropySGD(OptimizationFunction*, unsigned int, unsigned int, unsigned int,
    double, double, double, double, double, double, double, double);
  EntropySGD(unsigned int, OptimizationFunction*, Eigen::VectorXd);
  EntropySGD(unsigned int);
  EntropySGD(OptimizationFunction*, Eigen::VectorXd);
  EntropySGD(OptimizationFunction*);

  // set methods
  void set_parameters(unsigned int, unsigned int, unsigned int, double, double, double, double, double, double, double, double);

  // Overridden methods from abstract base class
  void solve() override;

  // get methods
  double get_epochs() const override;

};

#endif
