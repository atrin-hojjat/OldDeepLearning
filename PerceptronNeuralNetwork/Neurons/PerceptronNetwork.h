//
// Created by Atrin Hojjat on 5/30/16.
//

#ifndef PERCEPTRON_NEURAL_NETWORK_TEST_PERCEPTRONNETWORK_H
#define PERCEPTRON_NEURAL_NETWORK_TEST_PERCEPTRONNETWORK_H

#include "DEFINITIONS.h"
#include "PerceptronNeuron.h"

class PerceptronNetwork {
public:
    PerceptronNetwork(vector<__case> input,int first_layer_neurons,void (*listener)());

private :
    vector<double> *bias;
    Matrix *weight;
    vector<__case> *input;

    void (*listener)();

    int first_layer_nuerons_num;

    PerceptronNeuron **neurons;

    void setBaseValues();
    void start();
    void loop();
    void end();
};


#endif //PERCEPTRON_NEURAL_NETWORK_TEST_PERCEPTRONNETWORK_H
