#ifndef CNV_LAYER
#define CNV_LAYER

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <string>

class ConvLayer {
    private:
        int num_filters;
        int filter_size;
        std::vector<Eigen::MatrixXd> filters;
        std::vector<double> biases;

        Eigen::MatrixXd lastInputImage;
        std::vector<Eigen::MatrixXd> lastPreRelu; // Raw convolution results before ReLU implementation

    public:
        ConvLayer(int numFilters, int fSize);

        std::vector<Eigen::MatrixXd> forward(const Eigen::MatrixXd& inputImage);
        void backward(const std::vector<Eigen::MatrixXd>& dOutput, double learning_rate);

        void saveWeights(const std::string& filename);
        void loadWeights(const std::string& filename);

        //void printFilter() {std::cout << "ConvLayer Available Filter (Kernel):\n" << filters << "\n\n";}
};

#endif