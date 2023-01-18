#include <iostream>
#include <vector>

// Following https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

// Neuron Code
// Neurons are a function that returns the sum of a number of inputs, multiplied by their respective weights, and a bias
// Takes a vector of input values, a vector of weight values corresponding to those input values, and a bias
// Returns a single value of type double
// Every neuron in a layer takes in the same inputs, but has different weights and bias, and thus returns a different value

double neuron(std::vector<double> inputs, std::vector<double> weights, double bias)
{
    double output = 0;
    // sum each input by its corresponding weight
    for (int i = 0; i < inputs.size(); i++)
    {
        output += inputs[i] * weights[i];
    }

    return output + bias;
}

int main()
{
    // We know in advance the number of inputs and neurons per layer
    std::vector<double> inputs = {1, 2, 3, 2.5};
    std::vector<std::vector<double>> weights = {{0.2, 0.8, -0.5, 1.0}, {0.5, -0.91, 0.26, -0.5}, {-0.26, -0.27, 0.17, 0.87}};
    std::vector<double> biases = {2, 3, 0.5};

    for (int i = 0; i < weights.size(); i++)
    {
        double value = neuron(inputs, weights[i], biases[i]);
        std::cout << value << std::endl;
    }
}