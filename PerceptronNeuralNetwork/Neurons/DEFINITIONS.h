//
// Created by Atrin Hojjat on 5/30/16.
//

#ifndef PERCEPTRON_NEURAL_NETWORK_TEST_DEFINITIONS_H
#define PERCEPTRON_NEURAL_NETWORK_TEST_DEFINITIONS_H

#include <vector>

using namespace std;

typedef vector<vector<double>> Matrix;

struct __case {
    vector<double> input;
    vector<bool> output;
};

inline bool hardlim(double n){return n>=0;}



#endif //PERCEPTRON_NEURAL_NETWORK_TEST_DEFINITIONS_H
