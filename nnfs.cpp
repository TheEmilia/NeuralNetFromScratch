#include <iostream>
#include <numeric>
#include <vector>
#include <random>

using dynamic_matrix = std::vector<std::vector<double>>;
using dynamic_row = std::vector<double>;

// Following https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
// Using https://github.com/Sentdex/NNfSiX/tree/master/C%2B%2B for reference

// Function to generate spiral data set
// In python:
// def spiral_data(points, classes):
//     X = np.zeros((points*classes, 2))
//     y = np.zeros(points*classes, dtype='uint8')
//     for class_number in range(classes):
//         ix = range(points*class_number, points*(class_number+1))
//         r = np.linspace(0.0, 1, points)  # radius
//         t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
//         X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
//         y[ix] = class_number
//     return X, y
// Neuron: function that returns a dot product of inputs and weights summed with a bias
// Every neuron in a layer takes in the same inputs, but has different weights and bias, and thus returns different values

// Activation Functions: A function that changes a neuron's actual output based on the calculated output
// Step: return 1 if output>=0 else return 0
// ReLU: return output if output>=0 else return 0
// Sigmoid: return sigmoid of output (ie. 1/(1+e^-x))

// Used to print out the contents of a dynamic matrix
// Taken directly from https://github.com/Sentdex/NNfSiX/tree/master/C%2B%2B
std::ostream &operator<<(std::ostream &os, const dynamic_matrix &dm) noexcept
{
    for (auto &row : dm)
    {
        for (auto &item : row)
            os << item << " ";
        os << "\n";
    }
    return os;
}

// Generates a random double between a minimum and maximum value
// Taken directly from https://stackoverflow.com/a/35687575
template <typename Numeric, typename Generator = std::mt19937>
Numeric random(Numeric from, Numeric to)
{
    thread_local static Generator gen(std::random_device{}());

    using dist_type = typename std::conditional<
        std::is_integral<Numeric>::value, std::uniform_int_distribution<Numeric>, std::uniform_real_distribution<Numeric>>::type;

    thread_local static dist_type dist;

    return dist(gen, typename dist_type::param_type{from, to});
}

// Used to transpose a matrix - specifically weights before applying matrix multiplication in order to produce a correct output
// Modified from https://github.com/Sentdex/NNfSiX/tree/master/C%2B%2B
// No point in transposing weights - generate them to be the correct shape
dynamic_matrix transpose(const dynamic_matrix &input_matrix) noexcept
{
    // creates matrix to store transposition result
    dynamic_matrix output_matrix;

    // For each column in the input matrix
    for (int i = 0; i < input_matrix[0].size(); i++)
    {
        // insert an empty row for each inputted column
        output_matrix.push_back({});
        // for each row in the column
        for (int j = 0; j < input_matrix.size(); j++)
        {
            // insert into the output matrix at the current row the value for the next input from the input column
            output_matrix[i].push_back(input_matrix[j][i]);
        }
    }
    return output_matrix;
}

// matrix multiplication
// Modified from https://github.com/Sentdex/NNfSiX/tree/master/C%2B%2B
dynamic_matrix operator*(const dynamic_matrix &inputs_matrix, const dynamic_matrix &weights_matrix) noexcept
{
    // transposed version of the second matrix is created
    // dynamic_matrix weights_matrix = transpose(weights_matrix);
    // empty output matrix created
    dynamic_matrix output_matrix;

    // for each row in the first matrix
    for (int i = 0; i < inputs_matrix.size(); i++)
    {
        // create an empty row for the results
        output_matrix.push_back({});

        // for each column in the transposed matrix
        for (int j = 0; j < weights_matrix[0].size(); j++)
        {
            // result[row_a][column_b] = sum(matrix_a[row_a][k]*matrix_b[k][column_b]) for k=1 through n where n is the shared dimension
            double result = 0;

            // alternatively input_matrix[0].size()
            for (int k = 0; k < weights_matrix.size(); k++)
            {
                result += inputs_matrix[i][k] * weights_matrix[k][j];
            }

            // insert into the result matrix at the current column the dot product of the first matrix and the transposed matrix
            output_matrix[i].push_back(result);
        }
    }
    return output_matrix;
}

// matrix vector addition
// Modified from https://github.com/Sentdex/NNfSiX/tree/master/C%2B%2B
dynamic_matrix operator+(const dynamic_matrix &matrix, const dynamic_row &row) noexcept
{
    dynamic_matrix output_matrix;

    // for each row in the input matrix
    for (int j = 0; j < matrix.size(); j++)
    {
        // insert an empty row in the output matrix
        output_matrix.push_back({});

        // for each column in the input matrix
        for (int i = 0; i < matrix[0].size(); i++)
        {
            // insert into the output matrix the input matrix's value plus the bias in the corresponding column
            output_matrix[j].push_back(matrix[j][i] + row[i]);
        }
    }
    return output_matrix;
}

// ReLU object
class activation_ReLU
{
private:
    dynamic_matrix output_matrix;

public:
    void forward(const dynamic_matrix &layer_matrix)
    {
        output_matrix = dynamic_matrix();
        // For each row in the input matrix
        for (int i = 0; i < layer_matrix.size(); i++)
        {
            // insert an empty row for each inputted column
            output_matrix.push_back({});
            // for each column in the row
            for (int j = 0; j < layer_matrix[0].size(); j++)
            {
                double result = 0;
                if (layer_matrix[i][j] > 0)
                {
                    result = layer_matrix[i][j];
                }
                // insert into the output matrix at the current row the value for the next input from the input column
                output_matrix[i].push_back(result);
            }
        }
    }

    dynamic_matrix output() const
    {
        return output_matrix;
    }
};

// Taken directly from https://github.com/Sentdex/NNfSiX/tree/master/C%2B%2B
// Either load in a model ie. saved weights and biases from pre-built model
// Or initialise weights and biases at random
// Biases to a non-zero number (can result in a "dead network" otherwise), weights to non-zero number between 1 and -1
// FIXME When class is used, terminal stops outputting anything
class dense_layer
{
private:
    dynamic_matrix matrix_weights, matrix_output;
    dynamic_row matrix_biases;

public:
    // constructor for the layer
    // Takes in a number of inputs, a number of neurons, and then creates a matrix of weights which then are initialised
    dense_layer(const int &number_inputs, const int &number_neurons)
        : matrix_weights(number_inputs, dynamic_row(number_neurons)),
          matrix_biases(number_neurons, 1.0)
    {
        // For each
        for (int j = 0; j < number_neurons; j++)
        {
            for (int i = 0; i < number_inputs; i++)
                matrix_weights[i][j] = (random<double>(-1, 1));
        }
    }

    // pass forward the output of this layer
    void forward(const dynamic_matrix &inputs)
    {
        matrix_output = inputs * matrix_weights + matrix_biases;
    }

    dynamic_matrix output() const
    {
        return matrix_output;
    }
};

dynamic_matrix fixed_parameters(const dynamic_matrix &inputs, dynamic_matrix &matrix_weights, dynamic_row &matrix_biases)
{
    dynamic_matrix matrix_output = inputs * matrix_weights + matrix_biases;
    return matrix_output;
}

int main()
{

    // Inputs denoted by X, as per ML standards
    dynamic_matrix X{
        dynamic_row{1.0, 2.0, 3.0, 2.5},
        dynamic_row{2.0, 5.0, -1.0, 2.0},
        dynamic_row{-1.5, 2.7, 3.3, -0.8}};

    dynamic_matrix weights{
        dynamic_row{0.2, 0.5, -0.26},
        dynamic_row{0.8, -0.91, -0.27},
        dynamic_row{-0.5, 0.26, 0.17},
        dynamic_row{1.0, -0.5, 0.87}};

    dynamic_row biases = {2.0, 3.0, 0.5};
    dynamic_matrix test = fixed_parameters(X, weights, biases);
    std::cout << "\n"
              << test;

    dense_layer l1(4, 5);
    l1.forward(X);
    std::cout << "\n"
              << l1.output();

    activation_ReLU relu;
    relu.forward(l1.output());
    std::cout << "\n"
              << relu.output();

    dense_layer l2(5, 4);
    l2.forward(relu.output());
    std::cout << "\n"
              << l2.output();

    relu.forward(l2.output());
    std::cout << "\n"
              << relu.output();
}