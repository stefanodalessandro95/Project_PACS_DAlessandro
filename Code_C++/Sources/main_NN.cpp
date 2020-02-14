#include <stdlib.h>
#include <iostream>
#include "../Headers/OptimizationFunction.h"
#include "../Headers/FunctionData1D.h"
#include "../Headers/MinimizationAlgorithm.h"
#include "../Headers/StochGradDesc.h"
#include "../Headers/EntropySGD.h"
#include "../Headers/Heat.h"
#include "../Headers/GradientDescent.h"
#include "../Headers/FunctionOnNeuralNetwork.h"
#include <opennn.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <chrono>

int main(){
  unsigned int select_method;
  std::cout << "Choose the method for the solution: " << std::endl;
  std::cout << "1 -> SGD \n2 -> EntropySGD \n3 -> Heat \n4 -> all" << std::endl;
  std::cin >> select_method;
  std::cout << std::endl;

  // Dataset - MNIST
  OpenNN::DataSet mnist_data("/home/ste/Desktop/PACS/project/Code_C/Data/mnist_1200.csv", ',', false);

  mnist_data.set_input();
  mnist_data.set_column_use(0, OpenNN::DataSet::VariableUse::Target);
  mnist_data.numeric_to_categorical(0);

  const OpenNN::Vector<std::size_t> inputs_dimensions({1, 28, 28});
  const OpenNN::Vector<std::size_t> targets_dimensions({10});
  const std::size_t outputs_number = 10;
  mnist_data.set_input_variables_dimensions(inputs_dimensions);
  mnist_data.set_target_variables_dimensions(targets_dimensions);

  mnist_data.split_instances_random(0.9,0.0,0.1);
  std::cout << "Dataset made" << std::endl;
  std::cout << mnist_data.get_training_instances_number() << " training instances" << std::endl;
  std::cout << mnist_data.get_testing_instances_number() << " testing instances" << std::endl << std::endl;

  // Neural Network
  OpenNN::NeuralNetwork mnist_network;

  // Scaling layer
  OpenNN::ScalingLayer* scaling_layer = new OpenNN::ScalingLayer(inputs_dimensions);
  mnist_network.add_layer(scaling_layer);
  const OpenNN::Vector<size_t> scaling_layer_outputs_dimensions = scaling_layer->get_outputs_dimensions();

  // Convolutional layer 1
  OpenNN::ConvolutionalLayer* convolutional_layer_1 = new OpenNN::ConvolutionalLayer(scaling_layer_outputs_dimensions, {8, 5, 5});
  mnist_network.add_layer(convolutional_layer_1);
  const OpenNN::Vector<size_t> convolutional_layer_1_outputs_dimensions = convolutional_layer_1->get_outputs_dimensions();

  // Pooling layer 1
  OpenNN::PoolingLayer* pooling_layer_1 = new OpenNN::PoolingLayer(convolutional_layer_1_outputs_dimensions);
  mnist_network.add_layer(pooling_layer_1);
  const OpenNN::Vector<size_t> pooling_layer_1_outputs_dimensions = pooling_layer_1->get_outputs_dimensions();

  // Convolutional layer 2
  OpenNN::ConvolutionalLayer* convolutional_layer_2 = new OpenNN::ConvolutionalLayer(pooling_layer_1_outputs_dimensions, {4, 3, 3});
  mnist_network.add_layer(convolutional_layer_2);
  const OpenNN::Vector<size_t> convolutional_layer_2_outputs_dimensions = convolutional_layer_2->get_outputs_dimensions();

  // Pooling layer 2
  OpenNN::PoolingLayer* pooling_layer_2 = new OpenNN::PoolingLayer(convolutional_layer_2_outputs_dimensions);
  mnist_network.add_layer(pooling_layer_2);
  const OpenNN::Vector<size_t> pooling_layer_2_outputs_dimensions = pooling_layer_2->get_outputs_dimensions();

  // Convolutional layer 3
  OpenNN::ConvolutionalLayer* convolutional_layer_3 = new OpenNN::ConvolutionalLayer(pooling_layer_2_outputs_dimensions, {2, 3, 3});
  mnist_network.add_layer(convolutional_layer_3);
  const OpenNN::Vector<size_t> convolutional_layer_3_outputs_dimensions = convolutional_layer_3->get_outputs_dimensions();

  // Pooling layer 3
  OpenNN::PoolingLayer* pooling_layer_3 = new OpenNN::PoolingLayer(convolutional_layer_3_outputs_dimensions);
  mnist_network.add_layer(pooling_layer_3);
  const OpenNN::Vector<size_t> pooling_layer_3_outputs_dimensions = pooling_layer_3->get_outputs_dimensions();

  // Perceptron layer
  OpenNN::PerceptronLayer* perceptron_layer = new OpenNN::PerceptronLayer(pooling_layer_3_outputs_dimensions.calculate_product(), 18);
  mnist_network.add_layer(perceptron_layer);
  const size_t perceptron_layer_outputs = perceptron_layer->get_neurons_number();

  // Probabilistic layer
  OpenNN::ProbabilisticLayer* probabilistic_layer = new OpenNN::ProbabilisticLayer(perceptron_layer_outputs, outputs_number);
  mnist_network.add_layer(probabilistic_layer);

  std::cout << "Neural network information" << std::endl;
  mnist_network.print_summary();
  std::cout<<"Number of parameters: " << mnist_network.get_parameters_number() << std::endl << std::endl;

  // LossIndex
  OpenNN::MeanSquaredError* mnist_loss = new OpenNN::MeanSquaredError();

  FunctionOnNeuralNetwork mnist_function(mnist_loss);
  mnist_function.set_network_pointer(&mnist_network);
  mnist_function.set_dataset_pointer(&mnist_data);

  OpenNN::Vector<double> start = mnist_function.get_neural_network_pointer()->get_parameters();
  Eigen::VectorXd starting_point = Eigen::Map<Eigen::VectorXd>(start.data(), start.size());

  auto start_time = std::chrono::high_resolution_clock::now();
  auto stop_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time-start_time);

  if(select_method == 1 || select_method == 4){
    // Stochastic Gradient Descent
    std::cout << "SGD algorithm on the network" << std::endl;
    std::cout << "Starting function value: " << mnist_function.evaluate(starting_point) << std::endl;
    StochGradDesc SGD_on_mnist(&mnist_function,starting_point,250,60,0.0001,0.0001,0.6,10.0);
    start_time = std::chrono::high_resolution_clock::now();
    SGD_on_mnist.solve();
    stop_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time-start_time);
    std::cout << "Performed " << SGD_on_mnist.get_iterations() << " iterations" << std::endl;
    std::cout << "Total computation time: " << duration.count()/1000.0 << "s" << std::endl;
    std::cout << "Final function value: " << SGD_on_mnist.get_final_value() << std::endl << std::endl;
    Eigen::VectorXd opt_parameters_SGD = SGD_on_mnist.get_min();
    OpenNN::Vector<double> opt_parameters_v_SGD(opt_parameters_SGD.data(),
     opt_parameters_SGD.data() + opt_parameters_SGD.size());
    mnist_network.set_parameters(opt_parameters_v_SGD);

    OpenNN::TestingAnalysis testing_mnist_SGD(&mnist_network,&mnist_data);
    OpenNN::Matrix<size_t> confusion_mnist_SGD = testing_mnist_SGD.calculate_confusion();
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << confusion_mnist_SGD << std::endl;
    std::cout << "Accuracy: " << (confusion_mnist_SGD.calculate_trace()/confusion_mnist_SGD.calculate_sum())*100 << "%" << std::endl << std::endl;

    OpenNN::Vector<double> errors_SGD = testing_mnist_SGD.calculate_testing_errors();
    std::cout << "Mean Squared Error on sequestered data: " << errors_SGD[2] << std::endl;
  }

  if(select_method == 2 || select_method == 4){
    // EntropySGD
    std::cout << "EntropySGD algorithm on the network" << std::endl;
    std::cout << "Starting function value: " << mnist_function.evaluate(starting_point) << std::endl;
    EntropySGD EntropySGD_on_mnist(&mnist_function,starting_point,1,60,10,0.001,100.0,1.0,0.01,2.0,8.0,0.9,1.0);
    start_time = std::chrono::high_resolution_clock::now();
    EntropySGD_on_mnist.solve();
    stop_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time-start_time);
    std::cout << "Performed " << EntropySGD_on_mnist.get_iterations() << " iterations" << std::endl;
    std::cout << "Total computation time: " << duration.count()/1000.0 << "s" << std::endl;
    std::cout << "Final function value: " << EntropySGD_on_mnist.get_final_value() << std::endl << std::endl;
    Eigen::VectorXd opt_parameters_EntropySGD = EntropySGD_on_mnist.get_min();
    OpenNN::Vector<double> opt_parameters_v_EntropySGD(opt_parameters_EntropySGD.data(),
      opt_parameters_EntropySGD.data() + opt_parameters_EntropySGD.size());
    mnist_network.set_parameters(opt_parameters_v_EntropySGD);

    OpenNN::TestingAnalysis testing_mnist_EntropySGD(&mnist_network,&mnist_data);
    OpenNN::Matrix<size_t> confusion_mnist_EntropySGD = testing_mnist_EntropySGD.calculate_confusion();
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << confusion_mnist_EntropySGD << std::endl;
    std::cout << "Accuracy: " << (confusion_mnist_EntropySGD.calculate_trace()/confusion_mnist_EntropySGD.calculate_sum())*100 << "%" << std::endl << std::endl;

    OpenNN::Vector<double> errors_EntropySGD = testing_mnist_EntropySGD.calculate_testing_errors();
    std::cout << "Mean Squared Error on sequestered data: " << errors_EntropySGD[2] << std::endl;
  }

  if(select_method == 3 || select_method == 4){
    // Heat
    std::cout << "Heat algorithm on the network" << std::endl;
    std::cout << "Starting function value: " << mnist_function.evaluate(starting_point) << std::endl;
    Heat Heat_on_mnist(&mnist_function,starting_point,20,60,20,0.0001,2,0.5,10.0);
    start_time = std::chrono::high_resolution_clock::now();
    Heat_on_mnist.solve();
    stop_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time-start_time);
    std::cout << "Performed " << Heat_on_mnist.get_iterations() << " iterations" << std::endl;
    std::cout << "Total computation time: " << duration.count()/1000.0 << "s" << std::endl;
    std::cout << "Final function value: " << Heat_on_mnist.get_final_value() << std::endl << std::endl;
    Eigen::VectorXd opt_parameters_Heat = Heat_on_mnist.get_min();
    OpenNN::Vector<double> opt_parameters_v_Heat(opt_parameters_Heat.data(),
      opt_parameters_Heat.data() + opt_parameters_Heat.size());
    mnist_network.set_parameters(opt_parameters_v_Heat);

    OpenNN::TestingAnalysis testing_mnist_Heat(&mnist_network,&mnist_data);
    OpenNN::Matrix<size_t> confusion_mnist_Heat = testing_mnist_Heat.calculate_confusion();
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << confusion_mnist_Heat << std::endl;
    std::cout << "Accuracy: " << (confusion_mnist_Heat.calculate_trace()/confusion_mnist_Heat.calculate_sum())*100 << "%" << std::endl << std::endl;

    OpenNN::Vector<double> errors_Heat = testing_mnist_Heat.calculate_testing_errors();
    std::cout << "Mean Squared Error on sequestered data: " << errors_Heat[2] << std::endl;
  }

  return 0;
}
