//
// Created by Atrin Hojjat on 6/18/16.
//

#ifndef PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_SUPERVISEDHEBBIANNEURON_H
#define PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_SUPERVISEDHEBBIANNEURON_H

#include "DEFINITIONS.h"

class SupervisedHebbianNeuron {
public:
    SupervisedHebbianNeuron(){};
    SupervisedHebbianNeuron(MatrixXd *weight) : weight(weight){};

    void study(vector<__case>* wcase);
private:
    MatrixXd *weight;
};


#endif //PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_SUPERVISEDHEBBIANNEURON_H
