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
vector<double> inputs = {1, 2, 3, 2.5};
vector<vector<double>> weights = {{0.2, 0.8, -0.5, 1.0}, {0.5, -0.91, 0.26, -0.5}, {-0.26, -0.27, 0.17, 0.87}};
vector<double> biases = {2, 3, 0.5};

const double neuron(const vector<double> &weights, const vector<double> &inputs, double bias)
{
    return inner_product(weights.begin(), weights.end(), inputs.begin(), bias);
}

// Code adapted from https://stackoverflow.com/questions/61645527/how-can-i-do-a-dot-product-between-a-matrix-and-a-vector-in-c
// passing vectors by reference avoids unnecessary copies
vector<double> layer(const vector<vector<double>> &weights, const vector<double> &inputs, vector<double> &biases)
{
    vector<double> results;
    for (int i = 0; i < weights.size(); i++)
        results.push_back(neuron(weights[i], inputs, biases[i]));
    return results;
}

int main()
{
    vector<double> output = layer(weights, inputs, biases);

    for (auto &value : output)
        cout << value << endl;
}