#include "ConvLayer.h"
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>

Eigen::MatrixXd ConvLayer::forward(const Eigen::MatrixXd& inputImage) {
    this->lastInputImage = inputImage;

    int inputRows = inputImage.rows();
    int inputCols = inputImage.cols();

    // Output sizes
    int outputRows = inputRows - this->filter_size + 1;
    int outputCols = inputCols - this->filter_size + 1;

    // Feature map
    Eigen::MatrixXd outputMap(outputRows, outputCols);

    this->lastPreRelu = Eigen::MatrixXd(outputRows,outputCols);

    // Convolution (Sliding Window)
    for (int i = 0; i < outputRows; i++) {
        for (int j = 0; j < outputCols; j++) {
            Eigen::MatrixXd window = inputImage.block(i, j, this->filter_size, this->filter_size);

            double convolutionSum = (window.cwiseProduct(filter)).sum() + bias;

            lastPreRelu(i, j) = convolutionSum;

            // ReLU activation function: max(0, x)
            outputMap(i, j) = std::max(0.0, convolutionSum);
        }
    }

    return outputMap;
}

void ConvLayer::backward(const Eigen::MatrixXd& dOutput, double learning_rate) {
    int outRows = dOutput.rows();
    int outCols = dOutput.cols();

    // 1. Derivation of ReLU (It is 1 if it is greater than 0. Else, it is 0)
    Eigen::MatrixXd dRelu = dOutput;

    for (int i = 0; i < outRows; i++) {
        for (int j = 0; j < outCols; j++) {
            if (this->lastPreRelu(i, j) <= 0) {
                dRelu(i, j) = 0.0; // Those which don't pass ReLU won't have their mistakes counted either.
            }
        }
    }

    // 2. Bias and Filter Gradients are counted
    Eigen::MatrixXd dFilter = Eigen::MatrixXd::Zero(this->filter_size, this->filter_size);
    double dBias = dRelu.sum();

    // Filter gradient = Convolution of the original image with dReLU (error matrix).
    for (int i = 0; i < filter_size; i++) {
        for (int j = 0; j < filter_size; j++) {
            Eigen::MatrixXd window = lastInputImage.block(i, j, outRows, outCols);
            dFilter(i, j) = (window.cwiseProduct(dRelu)).sum();
        }
    }

    this->filter -= learning_rate * dFilter;
    this->bias -= learning_rate * dBias;
}