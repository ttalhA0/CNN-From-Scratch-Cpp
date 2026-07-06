#ifndef MAX_POOL_LAYER
#define MAX_POOL_LAYER

#include <Eigen/Dense>

class MaxPoolLayer {
    private:
        int poolSize;
        int stride; // how many pixels will the filter skip?

        // Memory for backpropagation
        Eigen::MatrixXi maxRowIndices;
        Eigen::MatrixXi maxColIndices;

        int lastInputRows;
        int lastInputCols;

    public:
        MaxPoolLayer(int pSize, int s) : poolSize(pSize), stride(s) {}
        Eigen::MatrixXd forward(const Eigen::MatrixXd& inputMap);
        Eigen::MatrixXd backward(const Eigen::MatrixXd& dOutput);
};

#endif