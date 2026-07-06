#ifndef MY_FILEREADER
#define MY_FILEREADER

#include <vector>
#include <string>
#include <Eigen/Dense>

void read_mnist_csv(const std::string& filename, std::vector<Eigen::MatrixXd>& X, std::vector<Eigen::VectorXd>& Y);

#endif