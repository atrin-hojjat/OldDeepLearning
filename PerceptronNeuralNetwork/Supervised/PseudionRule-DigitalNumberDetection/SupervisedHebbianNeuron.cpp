//
// Created by Atrin Hojjat on 6/18/16.
//

#include "SupervisedHebbianNeuron.h"

void SupervisedHebbianNeuron::study(vector<__case> *wcase)
{
    MatrixXd P(wcase->begin()->input.rows(),wcase->size()),T(wcase->begin()->output.rows(),wcase->size());

    for (int i = 0; i < wcase->size(); ++i) {
        P.col(i) = wcase->at(i).input;
        T.col(i) = wcase->at(i).output;
    }

    MatrixXd P_Inner_Product=P.transpose()*P;
    MatrixXd W;
    if(P_Inner_Product == MatrixXd::Identity(P_Inner_Product.rows(),P_Inner_Product.rows())){
        for (int i = 0; i < wcase->size(); ++i) {
            wcase->at(i).input.normalize();
            wcase->at(i).output.normalize();
            P.col(i) = wcase->at(i).input;
            T.col(i) = wcase->at(i).output;
        }
        W = T * P.transpose();
    } else {
        MatrixXd P_plus = (P.transpose() * P).inverse() * P.transpose();
        W = T * P_plus;
    }

    (*weight)=W;
}