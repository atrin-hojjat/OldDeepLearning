//
// Created by Atrin Hojjat on 7/5/16.
//

#ifndef PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_OPENCLCOSTFUNCTIONROUTER_H
#define PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_OPENCLCOSTFUNCTIONROUTER_H

#include <vector>
#include <functional>
#include <iomanip>
#include <iostream>
#include <OpenCL/opencl.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

using namespace std;

class OpenCLCostFunctionRouter {
public :
    OpenCLCostFunctionRouter(vector<vector<double>> input,vector<vector<double>> output,double learning_rate,double regularization_prameter);
    OpenCLCostFunctionRouter(vector<vector<double>> input,vector<vector<double>> output,double learning_rate,double regularization_prameter,long max_iters);

    function<vector<double>(vector<double>)> getTestFunc(){return this->test_func;}

    vector<double>getIterationStats(){return iter_stats;}
    vector<vector<double>> getWeightsMatrix(){return weight;}
private :

    vector<double> iter_stats;
    vector<vector<double>> weight;
    vector<vector<double>> input;
    vector<vector<double>> output;
    function<vector<double>(vector<double>)> test_func;
    long max_iters = -1;
    double learning_rate;
    double regularization_parameter;

protected:

    void call_func();
};


#endif //PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_OPENCLCOSTFUNCTIONROUTER_H
