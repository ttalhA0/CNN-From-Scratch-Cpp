#ifndef MAX_POOL_LAYER
#define MAX_POOL_LAYER

#include <Eigen/Dense>
#include <vector>

class MaxPoolLayer {
    private:
        int poolSize;
        int stride; // how many pixels will the filter skip?

        // Memory for backpropagation
        std::vector<Eigen::MatrixXi> maxRowIndices;
        std::vector<Eigen::MatrixXi> maxColIndices;

        int lastInputRows;
        int lastInputCols;

    public:
        MaxPoolLayer(int pSize, int s) : poolSize(pSize), stride(s) {}
        std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& inputMaps);
        std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& dOutput);
};

#endif