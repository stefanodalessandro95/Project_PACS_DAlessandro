#include "../Headers/OptimizationFunction.h"
#include "../Headers/FunctionOnNeuralNetwork.h"
#include <opennn.h>
#include <Eigen/Dense>
#include <stdlib.h>
#include <random>

// Constructors
FunctionOnNeuralNetwork::FunctionOnNeuralNetwork(OpenNN::LossIndex* loss_index_ptr):loss_function(loss_index_ptr){}

// Set methods
/// Sets the pointer to an exisitng OpenNN::NeuralNetwork
void FunctionOnNeuralNetwork::set_network_pointer(OpenNN::NeuralNetwork* new_network){
  loss_function->set_neural_network_pointer(new_network);
  set_state_dim(new_network->get_parameters_number());
}

/// Sets the pointer to an existing OpenNN::DataSet
void FunctionOnNeuralNetwork::set_dataset_pointer(OpenNN::DataSet* new_data){
  loss_function->set_data_set_pointer(new_data);
  data_dimention = new_data->get_training_instances_number();
}

// Overridden pure virtual methods
/// Evaluates the Loss index function for the passed values for the parameters of the neural network
double FunctionOnNeuralNetwork::evaluate(const Eigen::VectorXd &parameters_eigen) {
  std::vector<double> parameters_std(parameters_eigen.data(), parameters_eigen.data() + parameters_eigen.size());
  OpenNN::Vector<double> parameters = parameters_std;
  return loss_function->calculate_training_error(parameters);
}

/// Returns the stochastic gradient using the tools of the OpenNN library
Eigen::VectorXd FunctionOnNeuralNetwork::stochastic_gradient(const Eigen::VectorXd &parameters,unsigned int batch_dimention) {
  std::random_device rd; //seed
  std::default_random_engine gen(rd()); //engine
  std::uniform_int_distribution<> index(0, data_dimention-1); //distribution
  OpenNN::Vector<std::size_t> indices;
  for(unsigned int i=0; i<batch_dimention; i++)
    indices.push_back(index(gen));

  OpenNN::LossIndex::FirstOrderLoss first_order_loss = loss_function->calculate_batch_first_order_loss(indices);
  OpenNN::Vector<double> stoch_grad = first_order_loss.gradient;
  Eigen::VectorXd stoch_grad_eigen = Eigen::Map<Eigen::VectorXd>(stoch_grad.data(), stoch_grad.size());
  return stoch_grad_eigen;
}
