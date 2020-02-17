//
// Created by Atrin Hojjat on 7/13/16.
//

#ifndef AANN_NERUALNETWORK_H
#define AANN_NEURALNETWORK_H

#include <bitset>

using namespace std;

typedef bitset<8> byte;



class NeuralNetwork {
public:
    NeuralNetwork();
    NeuralNetwork(const byte* DATA,const int type);

protected:
    virtual void __learn__() const ;
    virtual void __init__() const ;
    virtual void __read_data__() const ;
    virtual void __visualize__() const ;

    const byte* DATA;
    const enum DATA_TYPE {

    } type;

    auto Test(const char* IN);

};


#endif //PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_NERUALNETWORK_H
