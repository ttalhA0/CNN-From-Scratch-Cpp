#include "MaxPoolLayer.h"
#include <Eigen/Dense>

std::vector<Eigen::MatrixXd> MaxPoolLayer::forward(const std::vector<Eigen::MatrixXd>& inputMaps) {
    int num_channel = inputMaps.size();
    this->lastInputRows = inputMaps[0].rows();
    this->lastInputCols = inputMaps[0].cols();

    int outRows = (this->lastInputRows - this->poolSize) / this->stride + 1;
    int outCols = (this->lastInputCols - this->poolSize) / this->stride + 1;

    std::vector<Eigen::MatrixXd> outputMaps(num_channel);

    // Initialize memory matrices at the output size.
    maxRowIndices.assign(num_channel, Eigen::MatrixXi(outRows, outCols));
    maxColIndices.assign(num_channel, Eigen::MatrixXi(outRows, outCols));

    for (int c = 0; c < num_channel; c++) {
        outputMaps[c] = Eigen::MatrixXd(outRows, outCols);

        for (int i = 0; i < outRows; i++) {
            int startRow = i * stride;
            for (int j = 0; j < outCols; j++) {
                int startCol = j * stride;

                Eigen::MatrixXd window = inputMaps[c].block(startRow, startCol, poolSize, poolSize);

                // Find the maximum value within the window and its LOCAL coordinates.
                Eigen::Index maxRow, maxCol;
                outputMaps[c](i, j) = window.maxCoeff(&maxRow, &maxCol);

                // Convert and save the local coordinates to the REAL coordinates in the original matrix.
                maxRowIndices[c](i, j) = startRow + maxRow;
                maxColIndices[c](i, j) = startCol + maxCol;
            }
        }
    }
    return outputMaps;
}

std::vector<Eigen::MatrixXd> MaxPoolLayer::backward(const std::vector<Eigen::MatrixXd>& dOutputs) {
    int num_channel = dOutputs.size();
    std::vector<Eigen::MatrixXd> dInputs(num_channel);

    int outRows = dOutputs[0].rows();
    int outCols = dOutputs[0].cols();

    for (int c = 0; c < num_channel; c++) {
        //  Create a zero matrix as an error matrix, using the original input size.
        dInputs[c] = Eigen::MatrixXd::Zero(lastInputRows, lastInputCols);

        // Place the errors only in the coordinates of the "winning" pixels.
        for (int i = 0; i < outRows; i++) {
            for (int j = 0; j < outCols; j++) {
                int originalRow = maxRowIndices[c](i, j);
                int originalCol = maxColIndices[c](i, j);
                
                // The error from MLP (dOutput) is being delivered to the correct address in memory.
                dInputs[c](originalRow, originalCol) = dOutputs[c](i, j);
            }
        }
    }
    return dInputs;
}