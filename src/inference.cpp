#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "fileReader.h"
#include "SimpleMLP.h"
#include "ConvLayer.h"
#include "MaxPoolLayer.h"
#include "ConfigParser.h"

#define IMAGE_WIDTH_LENGTH 28

int getArgument(int argc, char* argv[], std::string* dataset_path);

int main(int argc, char* argv[]) {
    std::string dataset_path = "data/mnist_test_750.csv"; // Default path

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

    std::vector<Eigen::MatrixXd> X_test;
    std::vector<Eigen::VectorXd> Y_test;

   std::cout << "--- DOWNLOADING TEST DATA ---\n";
    read_mnist_csv(dataset_path, X_test, Y_test);
    std::cout << "Test Size: " << X_test.size() << " images.\n\n";

    // Initiate layer objects
    ConvLayer conv(cfg.num_filters, cfg.filter_size); // The number of 32 3x3 filters for convolution layer 32 * [28x28 -> 26x26]
    MaxPoolLayer pool(cfg.pool_size, cfg.stride); // 2x2 pool, 2 step (stride) [26x26 -> 13x13]
    SimpleMLP mlp(cfg.flatten_size, cfg.hidden_nodes, cfg.output_nodes); // 169 input node, 64 secret node, 10 output

    std::cout << "--- LOADING PRE-TRAINED WEIGHTS ---\n";
    try {
        conv.loadWeights("conv_weights.bin");
        mlp.loadWeights("mlp_weights.bin");
        std::cout << "Weights loaded successfully!\n\n";
    } catch (const std::exception& e) {
        std::cerr << "Agirlik Yukleme Hatasi: " << e.what() << std::endl;
        return -1;
    }

    // For flattening
    int convRows = IMAGE_WIDTH_LENGTH - cfg.filter_size + 1;
    int poolRows = (convRows - cfg.pool_size) / cfg.stride + 1;
    int poolCols = poolRows;
    int mapSize = poolRows * poolCols;
    
    // 3. TESTING VE COMPUTING ACCURACY
    std::cout << "--- RUNNING INFERENCE ---\n";
    int correct_prediction = 0;

    for (size_t i = 0; i < X_test.size(); i++) {
        std::vector<Eigen::MatrixXd> convOut = conv.forward(X_test[i]);
        std::vector<Eigen::MatrixXd> poolOut = pool.forward(convOut);
        
        Eigen::VectorXd flattened(cfg.flatten_size);
        for (int f = 0; f < cfg.num_filters; f++) {
            // RowMajor read matrix
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rowMajorMap(poolOut[f]);
            Eigen::VectorXd flatMap = Eigen::Map<Eigen::VectorXd>(rowMajorMap.data(), rowMajorMap.size());

            // Place it in the relevant part of the vector.
            flattened.segment(f * mapSize, mapSize) = flatMap;
        }

        Eigen::VectorXd prediction = mlp.forward(flattened);

        Eigen::Index prediction_value, real_value;

        prediction.maxCoeff(&prediction_value);
        Y_test[i].maxCoeff(&real_value);

        if (prediction_value == real_value)
            correct_prediction++;

        // First 5 prediction is printed
        if (i < 5) {
            std::cout << "Image " << i+1 << " -> Prediction: " << prediction_value
                      << " | Real Value: " << real_value << std::endl;
        }
    }

    double accuracy = (double) correct_prediction / X_test.size() * 100.00;
    std::cout << "\n" << correct_prediction << " images are predicted correctly in total " << X_test.size() << " images." << std::endl;
    std::cout << "Accuracy: %" << accuracy << std::endl;

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