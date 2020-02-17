//
// Created by Atrin Hojjat on 5/30/16.
//

#include "PerceptronNeuron.h"

bool PerceptronNeuron::study(__case wcase)
{
    double n = *bias;
    for(int i =0;i<wcase.input.size();i++){
        n+=(*weight)[i]* wcase.input[i];
    }
    int a = hardlim(n) ? 1 :0;

    int e = wcase.output[num]-a;

    *bias+=e;
    for(int i =0;i<wcase.input.size();i++){
        (*weight)[i]+=e*wcase.input[i];
    }
    return e==0;
}