#include "ConvLayer.h"
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <stdexcept>

ConvLayer::ConvLayer(int numFilters, int fSize) : num_filters(numFilters), filter_size(fSize) {
    for (int i = 0; i < numFilters; i++) {
        filters.push_back(Eigen::MatrixXd::Random(fSize, fSize));
        biases.push_back(0.0);
    }
}

std::vector<Eigen::MatrixXd> ConvLayer::forward(const Eigen::MatrixXd& inputImage) {
    this->lastInputImage = inputImage;

    int inputRows = inputImage.rows();
    int inputCols = inputImage.cols();

    // Output sizes
    int outputRows = inputRows - this->filter_size + 1;
    int outputCols = inputCols - this->filter_size + 1;

    // Feature maps
    std::vector<Eigen::MatrixXd> outputMaps;
    this->lastPreRelu.clear();

    // Convolution (Sliding Window)
    for (int f = 0; f < num_filters; f++) {
        Eigen::MatrixXd outputMap(outputRows, outputCols);
        Eigen::MatrixXd preReluMap(outputRows, outputCols);
        
        for (int i = 0; i < outputRows; i++) {
            for (int j = 0; j < outputCols; j++) {
                Eigen::MatrixXd window = inputImage.block(i, j, this->filter_size, this->filter_size);

                double convolutionSum = (window.cwiseProduct(filters[f])).sum() + biases[f];

                preReluMap(i, j) = convolutionSum;

                // ReLU activation function: max(0, x)
                outputMap(i, j) = std::max(0.0, convolutionSum);
            }
        }

        this->lastPreRelu.push_back(preReluMap);
        outputMaps.push_back(outputMap);
    }

    return outputMaps;
}

void ConvLayer::backward(const std::vector<Eigen::MatrixXd>& dOutput, double learning_rate) {
    int outRows = dOutput[0].rows();
    int outCols = dOutput[0].cols();

    for (int f = 0; f < num_filters; f++) {
        // 1. Derivation of ReLU (It is 1 if it is greater than 0. Else, it is 0)
        Eigen::MatrixXd dRelu = dOutput[f];

        for (int i = 0; i < outRows; i++) {
            for (int j = 0; j < outCols; j++) {
                if (this->lastPreRelu[f](i, j) <= 0) {
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

        this->filters[f] -= learning_rate * dFilter;
        this->biases[f] -= learning_rate * dBias;
    }
}

void ConvLayer::saveWeights(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Weights of the ConvLayer cannot be saved: " + filename);

    // Write Filters one by one
    for (int f = 0; f < this->num_filters; f++) {
        out.write(reinterpret_cast<const char*>(filters[f].data()), filters[f].size() * sizeof(double));
    }

    // Write Biases
    out.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(double));

    out.close();
}

void ConvLayer::loadWeights(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Weights of the ConvLayer cannot be loaded: " + filename);

    // Read Filters one by one
    for (int f = 0; f < this->num_filters; f++) {
        in.read(reinterpret_cast<char*>(filters[f].data()), filters[f].size() * sizeof(double));
    }
    
    // Read Biases
    in.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(double));

    in.close();
}