//
// Created by Atrin Hojjat on 6/18/16.
//

#ifndef PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_SUPERVISEDHEBBIANNETWORK_H
#define PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_SUPERVISEDHEBBIANNETWORK_H

#include "DEFINITIONS.h"
#include "SupervisedHebbianNeuron.h"

class SupervisedHebbianNetwork {
public:
    SupervisedHebbianNetwork(vector<__case> *input,int layer_num,vector<int> neurons_num,void (*listener)());

    vector<vector<MatrixXd*>> getWeight(){return weight;}

private :
    vector<vector<MatrixXd*>> weight;
    vector<__case> *input;

    void (*listener)();

    int layer_num;
    vector<int> neurons_num;


    SupervisedHebbianNeuron **neurons;

    void setBaseValues();
    void start();
    void end();
};


#endif //PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_SUPERVISEDHEBBIANNETWORK_H
