//
// Created by Atrin Hojjat on 6/26/16.
//

#ifndef PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_LEARN_H
#define PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_LEARN_H

#include <vector>
#include <functional>
#include <Eigen/Dense>
#include <iostream>
#include <vtkVersion.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkChartXY.h>
#include <vtkTable.h>
#include <vtkPlot.h>
#include <vtkFloatArray.h>
#include <vtkContextView.h>
#include <vtkContextScene.h>
#include <vtkPen.h>

using namespace std;
using namespace Eigen;

class Learn {
public :
    Learn(MatrixXd input,MatrixXd output,double learning_rate,double regularization_prameter);
    Learn(MatrixXd input,MatrixXd output,double learning_rate,double regularization_prameter,long max_iters);

    function<VectorXd(VectorXd)> getTestFunc(){return this->test_func;}

    vector<double>getIterationStats(){return iter_stats;}
    MatrixXd getWeightsMatrix(){return weight;}

private:
    void learn();
    void drawLearningPlot();

    function<VectorXd(VectorXd)> test_func;

    vector<double> iter_stats;

    double learning_rate;
    double regularization_parameter;
    MatrixXd input;
    MatrixXd output;
    MatrixXd weight;
    long max_iters = -1;
};


#endif //PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_LEARN_H
