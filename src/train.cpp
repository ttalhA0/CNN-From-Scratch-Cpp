#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "fileReader.h"
#include "SimpleMLP.h"
#include "ConvLayer.h"
#include "MaxPoolLayer.h"
#include "ConfigParser.h"

#define IMAGE_WIDTH_LENGTH 28

int getArgument(int argc, char* argv[], std::string* dataset_path);

int main(int argc, char* argv[]) {
    std::string dataset_path = "data/mnist_train_3000.csv"; // Default path

    if (getArgument(argc, argv, &dataset_path) == -1) {
        std::cerr << "Usage: " << argv[0] << " [-d test_datapath.csv]\n";
        return -1;
    }

    ConfigParser cfg;
    try {
        cfg.load("config.txt");
        std::cout << "--- CONFIG LOADED SUCCESSFULLY ---\n";
        std::cout << "Filters: " << cfg.num_filters << " | Epochs: " << cfg.epochs 
                  << " | LR: " << cfg.learning_rate << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR]: " << e.what() << std::endl;
        return -1;
    }

    std::vector<Eigen::MatrixXd> X_train;
    std::vector<Eigen::VectorXd> Y_train;

    std::cout << "--- DOWNLOADING DATASET ---\n";
    read_mnist_csv(dataset_path, X_train, Y_train);
    std::cout << "\n";

    // Initiate layer objects
    ConvLayer conv(cfg.num_filters, cfg.filter_size); // The number of 32 3x3 filters for convolution layer 32 * [28x28 -> 26x26]
    MaxPoolLayer pool(cfg.pool_size, cfg.stride); // 2x2 pool, 2 step (stride) [26x26 -> 13x13]
    SimpleMLP mlp(cfg.flatten_size, cfg.hidden_nodes, cfg.output_nodes); // 169 input node, 64 secret node, 10 output

    // For flattening
    int convRows = IMAGE_WIDTH_LENGTH - cfg.filter_size + 1;
    int poolRows = (convRows - cfg.pool_size) / cfg.stride + 1;
    int poolCols = poolRows;
    int mapSize = poolRows * poolCols;
    
    std::cout << "X train size : " << X_train.size() << std::endl;
    std::cout << "Y train size : " << Y_train.size() << std::endl;

    std::cout << "--- CNN TRAINING IS INITIALIZING ---\n";
    for (int epoch = 0; epoch < cfg.epochs; epoch++) {
        double total_error = 0.0;

        for (size_t i = 0; i < X_train.size(); i++) {
            // ================= FORWARD PROPAGATION =================
            std::vector<Eigen::MatrixXd> convOutput = conv.forward(X_train[i]);
            std::vector<Eigen::MatrixXd> poolOut = pool.forward(convOutput);

            // Flatten
            Eigen::VectorXd flattened(cfg.flatten_size);
            for (int f = 0; f < cfg.num_filters; f++) {
                // RowMajor read matrix
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rowMajorMap(poolOut[f]);
                Eigen::VectorXd flatMap = Eigen::Map<Eigen::VectorXd>(rowMajorMap.data(), rowMajorMap.size());

                // Place it in the relevant part of the vector.
                flattened.segment(f * mapSize, mapSize) = flatMap;
            }

            // Prediction
            Eigen::VectorXd prediction = mlp.forward(flattened);

            // Calculation of Error (MSE)
            Eigen::VectorXd currentError = prediction - Y_train[i];
            total_error += currentError.squaredNorm();

            // ================= BACKWARD PROPAGATION =================
            
            // 1. Train mlp and get the error (dFlattened)
            Eigen::VectorXd dFlattened = mlp.backward(flattened, Y_train[i], cfg.learning_rate);

            //Convert the Vector to 13x13 Matrices
            std::vector<Eigen::MatrixXd> dPoolOutput(cfg.num_filters);
            for (int f = 0; f < cfg.num_filters; f++) {
                Eigen::VectorXd flatMap  = dFlattened.segment(f * mapSize, mapSize);

                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dPoolMap(flatMap.data(), poolRows, poolCols);
                dPoolOutput[f] = dPoolMap;
            }

            // 3. Pass the error through the pooling layer (Only pass it to the winning pixels)
            std::vector<Eigen::MatrixXd> dConvOutput = pool.backward(dPoolOutput);

            // 4. Train the convolution layer (Update filters)
            conv.backward(dConvOutput, cfg.learning_rate);
        }

        
        std::cout << "Epoch: " << epoch << " \t| Average Error: " << total_error / X_train.size() << std::endl;
    }

    std::cout << "\n--- TRAINING COMPLETED ---\n";

    std::cout << "Saving weights to binary files...\n";
    try {
        conv.saveWeights("conv_weights.bin");
        mlp.saveWeights("mlp_weights.bin");
        std::cout << "Success! Model weights saved as 'conv_weights.bin' and 'mlp_weights.bin'.\n";
    } catch (const std::exception& e) {
        std::cerr << "SAVING ERROR: " << e.what() << std::endl;
    }

    return 0;
}

int getArgument(int argc, char* argv[], std::string* dataset_path) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-d" && i + 1 < argc)
            *dataset_path = argv[++i];
        else
            return -1;
    }
    return 0;
}