#include "fileReader.h"
#include <iostream>
#include <fstream>

void read_mnist_csv(const std::string& filename, std::vector<Eigen::MatrixXd>& X, std::vector<Eigen::VectorXd>& Y) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error (The file cannot be opened): " << filename << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line); // Skip the first line


    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        std::getline(ss, cell, ','); // Get first label
        int label = std::stoi(cell);

        Eigen::VectorXd y_vec = Eigen::VectorXd::Zero(10);
        y_vec(label) = 1.0;
        Y.push_back(y_vec);

        // Kalan 784 sütun okunur
        Eigen::MatrixXd x_mat(28, 28);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                std::getline(ss, cell, ',');
                double pixel_value = std::stod(cell);
                x_mat(i, j) = pixel_value / 255.0; // Normalization
            }
        }
        X.push_back(x_mat);
    }

    file.close();
    std::cout << filename << " is downloaded. Total sample: " << X.size() << std::endl;
}