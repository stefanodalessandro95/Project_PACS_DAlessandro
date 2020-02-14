#ifndef HAVE_NEURAL_NETWORK_H
#define HAVE_NEURAL_NETWORK_H

#include "../Headers/opennn_headers/opennn.h"
#include "../Headers/OptimizationFunction.h"

/// This class is derived from the abstract class OptimizationFunction.
/// In particular this class uses the classes defined in the open library opennn
/// for the implementation of neural networks.
class
FunctionOnNeuralNetwork: public OptimizationFunction{
  private:
    OpenNN::LossIndex* loss_function;
    unsigned int data_dimention;

  public:
    // Constructors
    FunctionOnNeuralNetwork()=default;
    FunctionOnNeuralNetwork(OpenNN::LossIndex*);

    // Set methods
    void set_network_pointer(OpenNN::NeuralNetwork*);
    void set_dataset_pointer(OpenNN::DataSet*);

    // Overridden pure virtual methods
    double evaluate(const Eigen::VectorXd &) override;
    Eigen::VectorXd stochastic_gradient(const Eigen::VectorXd &,unsigned int) override;
    /// Returns the dimention of the training dataset
    unsigned int get_data_dim() const override {return data_dimention;}

    // Get method
    /// Returns a pointer to the OpenNN::NeuralNetwork used
    OpenNN::NeuralNetwork* get_neural_network_pointer() const {return loss_function->get_neural_network_pointer();}
};

#endif
