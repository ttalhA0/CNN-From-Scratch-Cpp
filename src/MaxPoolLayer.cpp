#include "MaxPoolLayer.h"
#include <Eigen/Dense>

Eigen::MatrixXd MaxPoolLayer::forward(const Eigen::MatrixXd& inputMap) {
    this->lastInputRows = inputMap.rows();
    this->lastInputCols = inputMap.cols();
    int outRows = (inputMap.rows() - this->poolSize) / this->stride + 1;
    int outCols = (inputMap.cols() - this->poolSize) / this->stride + 1;

    Eigen::MatrixXd outputMap(outRows, outCols);

    // Initialize memory matrices at the output size.
    maxRowIndices = Eigen::MatrixXi(outRows, outCols);
    maxColIndices = Eigen::MatrixXi(outRows, outCols);

    for (int i = 0; i < outRows; i++) {
        int startRow = i * stride;
        for (int j = 0; j < outCols; j++) {
            int startCol = j * stride;

            Eigen::MatrixXd window = inputMap.block(startRow, startCol, poolSize, poolSize);

            // Find the maximum value within the window and its LOCAL coordinates.
            Eigen::Index maxRow, maxCol;
            outputMap(i, j) = window.maxCoeff(&maxRow, &maxCol);

            // Convert and save the local coordinates to the REAL coordinates in the original matrix.
            maxRowIndices(i, j) = startRow + maxRow;
            maxColIndices(i, j) = startCol + maxCol;
        }
    }

    return outputMap;
}

Eigen::MatrixXd MaxPoolLayer::backward(const Eigen::MatrixXd& dOutput) {
    //  Create a zero matrix as an error matrix, using the original input size.
    Eigen::MatrixXd dInput = Eigen::MatrixXd::Zero(lastInputRows, lastInputCols);

    int outRows = dOutput.rows();
    int outCols = dOutput.cols();

    // Place the errors only in the coordinates of the "winning" pixels.
    for (int i = 0; i < outRows; i++) {
        for (int j = 0; j < outCols; j++) {
            int originalRow = maxRowIndices(i, j);
            int originalCol = maxColIndices(i, j);
            
            // The error from MLP (dOutput) is being delivered to the correct address in memory.
            dInput(originalRow, originalCol) = dOutput(i, j);
        }
    }
    
    return dInput;
}