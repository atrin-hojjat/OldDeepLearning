//
// Created by Atrin Hojjat on 6/26/16.
//

#include "Learn.h"

Learn::Learn(MatrixXd input, MatrixXd output,double regularization_parameter)  : regularization_parameter(regularization_parameter),input(input),output(output)
{
    learn();
}


void Learn::learn()
{
    MatrixXd w = MatrixXd::Zero(output.rows(),input.rows());

    auto pround = [](MatrixXd mat,double percision){
        MatrixXd ret(mat.rows(),mat.cols());
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                ret(i,j) = round(mat(i,j)*(1/percision))*percision;
            }
        }
        return ret;
    };

    MatrixXd P_Inner_Product=input.transpose()*input;
    MatrixXd W;
    if(P_Inner_Product == MatrixXd::Identity(P_Inner_Product.rows(),P_Inner_Product.rows())){
        input.normalize();
        output.normalize();
        W = output * input.transpose();
    } else {
        MatrixXd P_plus = (input.transpose() * input + regularization_parameter * MatrixXd::Identity(input.rows(),input.rows())).inverse() * input.transpose();
        W = output * P_plus;
    }

    w = pround(w,1e-10);

    cout << w;

    auto test = [w](VectorXd input) -> VectorXd{
        VectorXd output;

        output = w*input;

        return output;
    };

    this->weight = w;

    this->test_func = test;
}