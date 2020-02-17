//
// Created by Atrin Hojjat on 5/30/16.
//

#include "PerceptronNetwork.h"
#include <iostream>


PerceptronNetwork::PerceptronNetwork(vector<__case> input, int first_layer_neurons, void (*listener)())
{
    this->input = &input;
    this->listener = listener;
    this->first_layer_nuerons_num = first_layer_neurons;
    this->setBaseValues();
    this->start();

}

void PerceptronNetwork::setBaseValues()
{
    int S=first_layer_nuerons_num,R=(*input)[0].input.size();

    this->bias = new vector<double>(S);
    this->weight = new Matrix(R);

    this->bias->assign(S,0);

    vector<double>base_weight;
    base_weight.assign(R,0);
    this->weight->assign(S,base_weight);
}
void PerceptronNetwork::start()
{
    neurons = new PerceptronNeuron*[first_layer_nuerons_num];

    for (int i=0;i<first_layer_nuerons_num;i++){
        neurons[i] =new PerceptronNeuron(&(*bias)[i],&(*weight)[i],i);
    }
    loop();
}
void PerceptronNetwork::loop()
{
    for (int i=0;i<first_layer_nuerons_num;i++) {
        for (; ;) {
            bool temp=true;
            for(int j=0;j<input->size();j++){
                temp=neurons[i]->study((*input)[j])&& temp;
            }
            if(temp){break;}
        }
    }
    end();
}
void PerceptronNetwork::end()
{
    cout << "Bias : ";
    for(int i=0;i<bias->size();i++){
        cout << (*bias)[i] << " ";
    }
    cout << endl << "Weight : \n";
    for(int i=0;i<weight->size();i++){
        for(int j=0;j<(*weight)[i].size();j++) {
            cout << (*weight)[i][j] << " ";
        }
        cout <<endl;
    }
}
