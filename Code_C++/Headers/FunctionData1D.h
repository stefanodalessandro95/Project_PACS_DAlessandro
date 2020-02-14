#ifndef HAVE_FUNCTION_DATA_1D_H
#define HAVE_FUNCTION_DATA_1D_H

#include "OptimizationFunction.h"
#include <Eigen/Dense>
#include <stdlib.h>
#include <functional>
#include <memory>
#include <vector>

/// This class is derived from the abstract class OptimizationFunction.
/// This function is based on a dataset made of scalar values (double),
/// and provides also a method for the evaluation of the gradient.
class
FunctionData1D: public OptimizationFunction{

private :
  unsigned int data_dim=0;
  std::shared_ptr<std::vector<double>> dataset=NULL;

  std::function<double(const Eigen::VectorXd &,double)> f_base;
  std::function<Eigen::VectorXd(const Eigen::VectorXd &,double)> df_base;
  bool set_f_already = false;
  bool set_df_already = false;

public :
  // Constructors
  FunctionData1D() = default;
  FunctionData1D(unsigned int state_dim):OptimizationFunction(state_dim){};
  FunctionData1D(unsigned int,
	std::function<double(const Eigen::VectorXd &,double)>, std::function<Eigen::VectorXd(const Eigen::VectorXd &,double)>);

  // Dataset self-maker
  void make_dataset(unsigned int, double, double);

  // Overridden methods from the abstract base class
  double evaluate(const Eigen::VectorXd &) override;
  Eigen::VectorXd stochastic_gradient(const Eigen::VectorXd &,unsigned int) override;

  // New method, the full gradient is present only for this derived class
  Eigen::VectorXd gradient(const Eigen::VectorXd &);

  // set methods
  void set_f(std::function<double(const Eigen::VectorXd &,double)>);
  void set_df(std::function<Eigen::VectorXd(const Eigen::VectorXd &,double)> );
  void set_dataset(FunctionData1D*);
  void set_dataset(std::shared_ptr<std::vector<double>>);

  // get methods
  std::shared_ptr<std::vector<double>> get_dataset() {return dataset;}
  unsigned int get_data_dim() const override {return data_dim;}

};

#endif
