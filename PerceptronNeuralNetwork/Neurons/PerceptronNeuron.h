//
// Created by Atrin Hojjat on 5/30/16.
//

#ifndef PERCEPTRON_NEURAL_NETWORK_TEST_PERCEPTRONNEURON_H
#define PERCEPTRON_NEURAL_NETWORK_TEST_PERCEPTRONNEURON_H

#include "DEFINITIONS.h"

class PerceptronNeuron {
public:
    PerceptronNeuron(){};
    PerceptronNeuron(double *bias,vector<double> *weight,int num) : bias(bias),weight(weight),num(num){};

    bool study(__case wcase);
private:
    double *bias;
    int num;
    vector<double> *weight;

};


#endif //PERCEPTRON_NEURAL_NETWORK_TEST_PERCEPTRONNEURON_H
