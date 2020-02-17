//
// Created by Atrin Hojjat on 6/18/16.
//

#ifndef PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_DEFINITIONS_H
#define PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_DEFINITIONS_H

#include <iostream>
#include <fstream>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <vector>

using namespace std;
using namespace Eigen;

inline short hardlim(double n){return n>=0 ? 1:0;}
inline short hardlims(double n){return n>=0 ? 1:-1;}


struct __case {
    VectorXd input;
    VectorXd output;

};


#endif //PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_DEFINITIONS_H
