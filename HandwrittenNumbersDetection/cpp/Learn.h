//
// Created by Atrin Hojjat on 7/24/16.
//

#ifndef BACKPROPAGATION_DEMO_LEARN_CPP_H
#define BACKPROPAGATION_DEMO_LEARN_CPP_H

#include <functional>
#include <Eigen/Dense>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>

using namespace Eigen;
using namespace std;


void __rand_init(MatrixXd*,int,int);
void __rand_init(VectorXd*,int);

struct _transport_func {
    _transport_func(function<VectorXd(VectorXd)> transport_function, function<VectorXd(VectorXd)> transport_derivative)
            : transport_function(transport_function),transport_derivative(transport_derivative){

    }
    function<VectorXd(VectorXd)> transport_function;
    function<VectorXd(VectorXd)> transport_derivative;

    MatrixXd DerivativeMatrix(VectorXd n){
        MatrixXd ret = MatrixXd::Zero(n.rows(),n.rows());
        VectorXd derivatives = transport_derivative(n);
        for (int i = 0; i < n.rows(); ++i) {
            ret(i,i) = derivatives(i);
        }
        return ret;
    }
};

struct _layer{
    MatrixXd weighs;
    VectorXd bias;
    _transport_func transport;
    int input_dimention_num;
    int output_dimention_num;
    _layer(_transport_func transport,int input_dimention_num,int output_dimention_num) : input_dimention_num(input_dimention_num),output_dimention_num(output_dimention_num),transport(transport) {
        __rand_init(&weighs,output_dimention_num,input_dimention_num);
        __rand_init(&bias,output_dimention_num);
    }
};

enum trans_funcs{
    hardlim = 0,
    sigmoid = 1,
    pureline = 2,
    relu = 3
};

_transport_func getTransportFunc(int);
namespace SDBP {    //Steepest Descent Backpropagation
    void __learn(MatrixXd, MatrixXd, double /* Alpha */, vector<_layer>, int,double, function<VectorXd(VectorXd)> *);
}

namespace VLBP {    //Variable Learning-rate Backpropagation

}

namespace CGBP {    //Conjugate Gradient Backprapogation

}

namespace LMBP {    //Levenberg-Marquardt Backpropagation

}


#endif //BACKPROPAGATION_DEMO_LEARN_CPP_H
