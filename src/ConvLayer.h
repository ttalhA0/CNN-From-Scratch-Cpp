#ifndef CNV_LAYER
#define CNV_LAYER

#include <iostream>
#include <Eigen/Dense>

class ConvLayer {
    private:
        Eigen::MatrixXd filter;
        double bias;
        int filter_size;

        Eigen::MatrixXd lastInputImage;
        Eigen::MatrixXd lastPreRelu; // Raw convolution results before ReLU implementation

    public:
        ConvLayer(int fSize) : filter(Eigen::MatrixXd::Random(fSize, fSize)), bias(0.0), filter_size(fSize) {}

        Eigen::MatrixXd forward(const Eigen::MatrixXd& inputImage);
        void backward(const Eigen::MatrixXd& dOutput, double learning_rate);

        void printFilter() {std::cout << "ConvLayer Available Filter (Kernel):\n" << filter << "\n\n";}
};

#endif