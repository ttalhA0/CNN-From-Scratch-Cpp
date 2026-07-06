#ifndef SIMPLE_MLP
#define SIMPLE_MLP

#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "fileReader.h"

/*
    The sigmoid function is used as a activation function

    Sigmoid func -> 1 / (1 + e^(-x))
*/

class SimpleMLP {
    private:
        // First Layer (Hidden Layer)
        Eigen::MatrixXd W1; // ilk katmanın ağırlıkları
        Eigen::VectorXd b1; // ilk katmanın bias'ları

        // Output layer
        Eigen::MatrixXd W2;
        Eigen::VectorXd b2;

        // Intermediate values ​​calculated in forward propagation and used in backpropagation.
        Eigen::VectorXd Z1;
        Eigen::VectorXd A1;
        Eigen::VectorXd Z2;
        Eigen::VectorXd A2;


        // Activation Fucntion (Sigmoid)
        static double sigmoid(double x) {
            return 1.0 / (1.0 + std::exp(-x));
        }

    public:
        // Constructor
        SimpleMLP (int inputNodes, int hiddenNodes, int outputNodes);

        Eigen::VectorXd forward(const Eigen::VectorXd& X);

        Eigen::VectorXd backward(const Eigen::VectorXd& X, const Eigen::VectorXd Y, double learning_rate);

        void train(const std::vector<Eigen::VectorXd>& X_train, const std::vector<Eigen::VectorXd>& Y_train, int epochs, double learning_rate);
};


#endif