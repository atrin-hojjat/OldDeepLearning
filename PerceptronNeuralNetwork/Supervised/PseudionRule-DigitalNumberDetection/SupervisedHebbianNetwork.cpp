//
// Created by Atrin Hojjat on 6/18/16.
//

#include "SupervisedHebbianNetwork.h"
SupervisedHebbianNetwork::SupervisedHebbianNetwork(vector<__case> *input,int layer_num,vector<int> neurons_num,void (*listener)())
        : input(input),layer_num(layer_num),neurons_num(neurons_num),listener(listener){
    setBaseValues();
    start();
}

void SupervisedHebbianNetwork::setBaseValues() {
    weight =  vector<vector<MatrixXd*>>(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        weight[i] = vector<MatrixXd*>(neurons_num[i],new MatrixXd((*input)[0].output.rows(),(*input)[0].input.cols()));
    }
}

void SupervisedHebbianNetwork::start() {
    neurons = new SupervisedHebbianNeuron*[layer_num];

    for (int i = 0; i < layer_num; ++i) {
        neurons[i] = new SupervisedHebbianNeuron[neurons_num[i]];
        for (int j = 0; j < neurons_num[i]; ++j) {
            neurons[i][j] = SupervisedHebbianNeuron(weight[i][j]);
        }
    }

    for (int i = 0; i < layer_num; ++i) {
        for (int j = 0; j < neurons_num[i]; ++j) {
            neurons[i][j].study(input);
        }
    }

    end();
}


void SupervisedHebbianNetwork::end() {

}