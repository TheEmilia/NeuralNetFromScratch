#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

// Following https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

// Neuron Code
// Neuron: function that returns the sum of a number of inputs, multiplied by their respective weights, and a bias
// Takes a vector of input values, a vector of weight values corresponding to those input values, and a bias
// Returns a single value of type double
// Every neuron in a layer takes in the same inputs, but has different weights and bias, and thus returns a different value

// We know in advance the number of inputs and neurons per layer
vector<vector<double>> inputs = {{1, 2, 3, 2.5}, {2.0, 5.0, -1.0, 2.0}, {-1.5, 2.7, 3.3, -0.8}};
vector<vector<double>> weights = {{0.2, 0.8, -0.5, 1.0}, {0.5, -0.91, 0.26, -0.5}, {-0.26, -0.27, 0.17, 0.87}};
vector<double> biases = {2, 3, 0.5};

// Used to transpose weights before applying matrix multiplication in order to produce a correct output
const vector<vector<double>> transposeMatrix(const vector<vector<double>> &input_matrix)
{
    // Number of inputs = number of weights per neuron
    // Number of neurons = number of batches of weights
    // Weight default: weights(x axis) by batches(y axis)
    // Transpose to batches(x) by weights(y)

    // Dimensions of input matrix
    int row_count = input_matrix.size();
    int column_count = input_matrix[0].size();

    // Check for suitable dimensions otherwise throw an error
    if (row_count <= 0 || column_count <= 0)
    {
        cout << "Shape Error, check dimensions of transposed matrix" << endl;
        return input_matrix;
    }

    // Create empty matrix of the correct dimensions
    vector<vector<double>> transposed_matrix(column_count, vector<double>(row_count, 0));

    // Swap rows with columns  and set value in transposed matrix
    for (int i = 0; i < column_count; ++i)
    {
        for (int j = 0; j < row_count; ++j)
        {
            transposed_matrix[i][j] = input_matrix[j][i];
        }
    }

    return transposed_matrix;
}

// Acts as layer composed of inputs. matrix1 should be inputs, matrix2 should be transposed weights, and biases offset the calculated value for the corresponding neuron
const vector<vector<double>> multiplyMatrices(const vector<vector<double>> &matrix_inputs, const vector<vector<double>> &matrix_weights, const vector<double> &biases)
{
    // Dimensions of inputs matrix
    int row_count_inputs = matrix_inputs.size();
    int column_count_inputs = matrix_inputs[0].size();

    // Dimensions of weights matrix
    int row_count_weights = matrix_weights.size();
    int column_count_weights = matrix_weights[0].size();

    // Check for suitable dimensions otherwise throw an error
    if (row_count_inputs <= 0 || column_count_inputs <= 0 || column_count_weights <= 0 || row_count_weights <= 0)
    {
        cout << "Shape Error, check dimensions of transposed matrix" << endl;
        return matrix_inputs;
    }
    if (column_count_inputs != row_count_weights)
    {
        cout << "Shape Error, check dimensions of transposed matrix" << endl;
        return matrix_inputs;
    }

    // Create empty matrix of the correct dimensions
    vector<vector<double>> output_matrix(row_count_inputs, vector<double>(column_count_weights, 0));

    for (int i = 0; i < row_count_inputs; ++i)
    {
        for (int j = 0; j < column_count_inputs; ++j)
        {
            output_matrix[i][j] = biases[j];
            cout << i << ", " << j << ", " << output_matrix[i][j] << endl;
            for (int k = 0; k < column_count_weights; ++k)
            {
                output_matrix[i][j] += matrix_inputs[i][k] * matrix_weights[k][j]; // CHeck to ensure correct multiplication is occurring
            }
            cout << i << ", " << j << ", " << output_matrix[i][j] << endl;
        }
    }

    return output_matrix;
}

// const vector<vector<double>> applyBiases(const vector<vector<double>> &matrix, const vector<double> &biases)
// {
//     // Dimensions of matrix - should always be square
//     int side_length = matrix.size();

//     vector<vector<double>> output_matrix = matrix;

//     // traverses left to right then top to bottom
//     for (int i = 0; i < side_length; ++i)
//     {
//         for (int j = 0; j < side_length; ++j)
//         {
//             output_matrix[i][j] += biases[j];
//         }
//     }

//     return output_matrix;
// }

void printMatrix(vector<vector<double>> input_matrix)
{

    int sides = input_matrix.size();

    for (int i = 0; i < sides; ++i)
    {
        for (int j = 0; j < sides; ++j)
        {
            cout << input_matrix[i][j] << " - ";
        }
        cout << endl;
    }
}

int main()
{
    const vector<vector<double>> transposed_weights = transposeMatrix(weights);
    const vector<vector<double>> output = multiplyMatrices(inputs, transposed_weights, biases);

    // vector<vector<double>> output = applyBiases(output, biases);
    // Biases still need to be applied
    // doesn't print output values when run
    cout << endl;
    cout << "TEST" << endl;
    // printMatrix(output);
}