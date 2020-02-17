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
    Learn(MatrixXd input,MatrixXd output,double regularization_prameter);

    function<VectorXd(VectorXd)> getTestFunc(){return this->test_func;}

    MatrixXd getWeightsMatrix(){return weight;}

private:
    void learn();

    function<VectorXd(VectorXd)> test_func;

    double regularization_parameter;
    MatrixXd input;
    MatrixXd output;
    MatrixXd weight;
};


#endif //PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_LEARN_H
