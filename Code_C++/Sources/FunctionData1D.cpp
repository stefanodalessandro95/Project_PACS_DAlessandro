#include "../Headers/OptimizationFunction.h"
#include "../Headers/FunctionData1D.h"
#include <Eigen/Dense>
#include <stdlib.h>
#include <iostream>
#include <functional>
#include <memory>
#include <random>

// Constructors
FunctionData1D::FunctionData1D(unsigned int state_dim,
	std::function<double(const Eigen::VectorXd &,double)> f, std::function<Eigen::VectorXd(const Eigen::VectorXd &,double)> df)
	:OptimizationFunction(state_dim),f_base(f),df_base(df){
		set_f_already = true;
		set_df_already = true;
	}

// Overridden methods from abstract base class
/// This method evaluates the function, by summing the value of the base function
/// on the dataset, according to the given value of the parameters (the argument x).
double FunctionData1D::evaluate(const Eigen::VectorXd &x) {
	if(!set_f_already){
		std::cout << "First the base function has to be set" << std::endl;
		return 0.0;
  	}

	if(x.size()!=get_state_dim()){
		std::cout << "Dimension of the point is not consistent with the one required" << std::endl;
		std::cout << "Dimension required = " << get_state_dim() << std::endl;
		return 0.0;
  	}

	if(dataset == NULL){
		std::cout << "The Dataset has not been bult yet" << std::endl;
		return 0.0;
  	}

	double result = 0.0;
	for(unsigned int i=0; i<data_dim; i++)
    		result+= f_base(x,(*dataset)[i]);
  	return result/data_dim;
}

/// Returns the value of the stochastic gradient at the point x given as a parameter.
/// The stochastic gradient is computed by summing the value of te gradient only for
/// some of the instances in the dataset.
Eigen::VectorXd FunctionData1D::stochastic_gradient(const Eigen::VectorXd &x, unsigned int mini_batch_size) {
	if(!set_df_already){
		std::cout << "First the base function gradient has to be set" << std::endl;
		return Eigen::VectorXd::Zero(get_state_dim());
  }

	if(x.size()!=get_state_dim()){
		std::cout << "Dimension of the point is not consistent with the one required" << std::endl;
		std::cout << "Dimension required = " << get_data_dim() << std::endl;
		return Eigen::VectorXd::Zero(get_state_dim());
	 }

	if(dataset == NULL){
		std::cout << "The Dataset has not been bult yet" << std::endl;
		return Eigen::VectorXd::Zero(get_state_dim());
	 }

	Eigen::VectorXd result = Eigen::VectorXd::Zero(get_state_dim());

  std::random_device rd; //seed
  std::default_random_engine gen(rd()); //engine
  std::uniform_int_distribution<> index(0, data_dim-1); //distribution

  for(unsigned int i=0; i<mini_batch_size; i++){
		unsigned int ind = index(gen);
    result+= df_base(x,(*dataset)[ind]);
	}

  return result/mini_batch_size;
}

/// This method is not overridden from the base class. It computes the whole gradient
/// by summing the gradient computed for ALL the instances in the dataset.
Eigen::VectorXd FunctionData1D::gradient(const Eigen::VectorXd &x) {
	if(!set_df_already){
		std::cout << "First the base function gradient has to be set" << std::endl;
		return Eigen::VectorXd::Zero(get_state_dim());;
  }

	if(x.size()!=get_state_dim()){
		std::cout << "Dimension of the point is not consistent with the one required" << std::endl;
		std::cout << "Dimension required = " << get_data_dim() << std::endl;
		return Eigen::VectorXd::Zero(get_state_dim());
	}

	if(dataset == NULL){
		std::cout << "Dataset not already built" << std::endl;
		return Eigen::VectorXd::Zero(get_state_dim());
  }


	Eigen::VectorXd result = Eigen::VectorXd::Zero(get_state_dim());
	for(unsigned int i = 0; i<data_dim; i++)
    		result+= df_base(x,(*dataset)[i]);
  	return result/data_dim;
}

/// This method builds the dataset according to the parameters: the number of elements,
/// the min and the max value. The dataset is build in a randomic way using a uniform distribution.
void FunctionData1D::make_dataset(unsigned int data_dimention, double data_min, double data_max){
	if(dataset.get() != NULL){
		dataset.reset();
		std::cout << "The previous dataset has been removed" << std::endl;
	}

	std::cout << "Dataset is being built... ";
	data_dim = data_dimention;
  dataset = std::make_shared<std::vector<double>>(*(new std::vector<double>));

  std::random_device rd; //seed
  std::default_random_engine re(rd()); //engine
  std::uniform_real_distribution<double> unif(data_min,data_max); //distribution

  for(unsigned int i = 1; i<=data_dim; i++)
    dataset->push_back((double)unif(re));

	std::cout << "Done" << std::endl;
}

// Set methods
/// Sets the base function
void FunctionData1D::set_f(std::function<double(const Eigen::VectorXd &,double)> f){
	f_base = f;
	set_f_already = true;
}

/// Sets the gradient of the base function
void FunctionData1D::set_df(std::function<Eigen::VectorXd(const Eigen::VectorXd &,double)> df){
	df_base = df;
	set_df_already = true;
}

/// Sets the dataset, by copying the pointer to the dataset of another function.
/// (The values are not copied)
void FunctionData1D::set_dataset(FunctionData1D* other_function){
	if(dataset.get() != NULL){
		dataset.reset();
		std::cout << "The previous dataset has been removed" << std::endl;
	}
	dataset = other_function->get_dataset();
}

/// Sets the dataset using an already existing dataset
void FunctionData1D::set_dataset(std::shared_ptr<std::vector<double>> new_dataset){
	dataset = new_dataset;
	data_dim = dataset->size();
}
