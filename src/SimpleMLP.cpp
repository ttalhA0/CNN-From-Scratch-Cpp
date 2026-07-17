#include "SimpleMLP.h"
#include <fstream>
#include <stdexcept>

SimpleMLP::SimpleMLP (int inputNodes, int hiddenNodes, int outputNodes) {
    // We initiate weights between -1 and 1 randomly.
    W1 = Eigen::MatrixXd::Random(hiddenNodes, inputNodes);
    W2 = Eigen::MatrixXd::Random(outputNodes, hiddenNodes);

    // We iniate biases with 0
    b1 = Eigen::VectorXd::Zero(hiddenNodes);
    b2 = Eigen::VectorXd::Zero(outputNodes);
}

Eigen::VectorXd SimpleMLP::forward(const Eigen::VectorXd& X) {
    //  Hidden layer calculation, also known as the convolutional layer.
    Z1 = W1 * X + b1;

    // The sigmoid function, which is the activation function for each element, is used.
    A1 = Z1.unaryExpr([](double x) {return sigmoid(x); });

    // Output Layer calculations
    Z2 = W2 * A1 + b2;
    A2 = Z2.unaryExpr([](double x) {return sigmoid(x); });

    return A2; // Prediction
}

Eigen::VectorXd SimpleMLP::backward(const Eigen::VectorXd& X, const Eigen::VectorXd Y, double learning_rate) {    
    //Output Layer Error and Delta (Prediction - Actual)
    Eigen::VectorXd error = A2 - Y;

    // delta2 = error * derivation(A2)
    Eigen::VectorXd delta2 = error.cwiseProduct(A2.unaryExpr([](double a) {return a * (1.0 - a); }));

    // Error and Delta for the hidden layer.
    // To find the error in the hidden layer, we multiply delta2 by the transpose of W2.
    Eigen::VectorXd error1 = W2.transpose() * delta2;

    // delta1 = error1 * derivation(A1)
    Eigen::VectorXd delta1 = error1.cwiseProduct(A1.unaryExpr([](double a) {return a * (1.0 - a); }));

    // Update Weights and Deviations
    // dW2 = delta2 * transpose of the A1
    W2 -= learning_rate * (delta2 * A1.transpose());
    b2 -= learning_rate * delta2;

    // dW1 = delta1 * transpose of the X
    W1 -= learning_rate * (delta1 * X.transpose());
    b1 -= learning_rate * delta1;

    Eigen::VectorXd dInput = W1.transpose() * delta1;
    return dInput;
}

void SimpleMLP::train(const std::vector<Eigen::VectorXd>& X_train, const std::vector<Eigen::VectorXd>& Y_train, int epochs, double learning_rate) {
    std::cout << "Training is being initiliazed" << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0.0;

        for (size_t i = 0; i < X_train.size(); i++) {
            Eigen::VectorXd prediction = this->forward(X_train[i]);

            this->backward(X_train[i], Y_train[i], learning_rate);

            // Calculate the mean squared error just for informational purposes.
            Eigen::VectorXd currentError = prediction - Y_train[i];
            total_error += currentError.squaredNorm(); // Add the squares of the errors.
        }

        if (epoch % 1000 == 0) {
            std::cout << "Epoch: " << epoch << " \t| Average Error (MSE): " << total_error / X_train.size() << std::endl;
        }
    }
    std::cout << "TRAINING IS COMPLETED\n\n";
}

void SimpleMLP::saveWeights(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Weights of the MLP cannot be saved: " + filename);
    }

    out.write(reinterpret_cast<const char*>(W1.data()), W1.size() * sizeof(double));
    out.write(reinterpret_cast<const char*>(b1.data()), b1.size() * sizeof(double));
    
    out.write(reinterpret_cast<const char*>(W2.data()), W2.size() * sizeof(double));
    out.write(reinterpret_cast<const char*>(b2.data()), b2.size() * sizeof(double));

    out.close();
}

void SimpleMLP::loadWeights(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("MLP agirliklari okunamadi: " + filename);
    }

    // When reading, it is crucial that the dimensions of the matrices are predefined in the constructor.
    // Otherwise, you will get a Segmentation Fault error.
    in.read(reinterpret_cast<char*>(W1.data()), W1.size() * sizeof(double));
    in.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(double));
    
    in.read(reinterpret_cast<char*>(W2.data()), W2.size() * sizeof(double));
    in.read(reinterpret_cast<char*>(b2.data()), b2.size() * sizeof(double));

    in.close();
}