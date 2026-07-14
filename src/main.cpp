#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "fileReader.h"
#include "SimpleMLP.h"
#include "ConvLayer.h"
#include "MaxPoolLayer.h"

int main() {
    std::vector<Eigen::MatrixXd> X_train, X_test;
    std::vector<Eigen::VectorXd> Y_train, Y_test;

   std::cout << "--- DOWNLOADING DATASET ---\n";
    read_mnist_csv("data/mnist_train_3000.csv", X_train, Y_train);
    read_mnist_csv("data/mnist_test_750.csv", X_test, Y_test);
    std::cout << "\n";

    // Initiate layer objects
    int num_filters = 32;
    ConvLayer conv(num_filters, 3); // The number of 32 3x3 filters for convolution layer 32 * [28x28 -> 26x26]
    MaxPoolLayer pool(2, 2); // 2x2 pool, 2 step (stride) [26x26 -> 13x13]
    SimpleMLP mlp(5408, 128, 10); // 169 input node, 64 secret node, 10 output

    double learning_rate = 0.01;
    int epochs = 50;

    // For flattening
    int poolRows = 13;
    int poolCols = 13;
    int mapSize = poolRows * poolCols;
    Eigen::VectorXd flattened(num_filters * mapSize);

    std::cout << "X train size : " << X_train.size() << std::endl;
    std::cout << "X test size : " << X_test.size() << std::endl;
    std::cout << "Y train size : " << Y_train.size() << std::endl;
    std::cout << "Y test size : " << Y_test.size() << std::endl;

    std::cout << "--- CNN TRAINING IS INITIALIZING ---\n";
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0.0;

        for (size_t i = 0; i < X_train.size(); i++) {
            // ================= FORWARD PROPAGATION =================
            std::vector<Eigen::MatrixXd> convOutput = conv.forward(X_train[i]);
            std::vector<Eigen::MatrixXd> poolOut = pool.forward(convOutput);

            // Flatten
            Eigen::VectorXd flattened(num_filters * mapSize);
            for (int f = 0; f < num_filters; f++) {
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
            Eigen::VectorXd dFlattened = mlp.backward(flattened, Y_train[i], learning_rate);

            //Convert the Vector to 13x13 Matrices
            std::vector<Eigen::MatrixXd> dPoolOutput(num_filters);
            for (int f = 0; f < num_filters; f++) {
                Eigen::VectorXd flatMap  = dFlattened.segment(f * mapSize, mapSize);

                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dPoolMap(flatMap.data(), poolRows, poolCols);
                dPoolOutput[f] = dPoolMap;
            }

            // 3. Pass the error through the pooling layer (Only pass it to the winning pixels)
            std::vector<Eigen::MatrixXd> dConvOutput = pool.backward(dPoolOutput);

            // 4. Train the convolution layer (Update filters)
            conv.backward(dConvOutput, learning_rate);
        }

        
        std::cout << "Epoch: " << epoch << " \t| Average Error: " << total_error / X_train.size() << std::endl;
    }

    std::cout << "\n--- TRAINING COMPLETED ---\n";

    
    // 3. TESTING VE COMPUTING ACCURACY
    std::cout << "--- EVALUATED BASED ON TEST DATA. (250 IMAGES) ---\n";
    int correct_prediction = 0;

    for (size_t i = 0; i < X_test.size(); i++) {
        std::vector<Eigen::MatrixXd> convOut = conv.forward(X_test[i]);
        std::vector<Eigen::MatrixXd> poolOut = pool.forward(convOut);
        
        Eigen::VectorXd flattened(num_filters * mapSize);
        for (int f = 0; f < num_filters; f++) {
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