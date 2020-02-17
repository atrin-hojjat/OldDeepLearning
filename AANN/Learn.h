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
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
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

using namespace Eigen;
using namespace std;
using namespace chrono;


void __rand_init(MatrixXd*,int,int);
void __rand_init(VectorXd*,int);

const auto __plot_productivity_function__ = [](vector<double> iter_stats) -> void{
    vtkSmartPointer<vtkTable> table =
            vtkSmartPointer<vtkTable>::New();

    vtkSmartPointer<vtkFloatArray> arrX =
            vtkSmartPointer<vtkFloatArray>::New();
    arrX->SetName("Number Of Iterations");
    table->AddColumn(arrX);

    vtkSmartPointer<vtkFloatArray> arrC =
            vtkSmartPointer<vtkFloatArray>::New();
    arrC->SetName("Cost Function");
    table->AddColumn(arrC);

    // Fill in the table with some example values
    int numPoints = iter_stats.size();
    table->SetNumberOfRows(numPoints);
    for (int i = 0; i < numPoints; ++i)
    {
        table->SetValue(i, 0, i);
        table->SetValue(i, 1, iter_stats[i]);
    }

    // Set up the view
    vtkSmartPointer<vtkContextView> view =
            vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);

    // Add multiple line plots, setting the colors etc
    vtkSmartPointer<vtkChartXY> chart =
            vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    vtkPlot *line = chart->AddPlot(vtkChart::POINTS);
    line->SetInputData(table, 0, 1);
    line->SetColor(1, 0, 0);
    line->SetWidth(2.0);
    line = chart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, 1);
    line->SetColor(0, 0, 1);
    line->SetWidth(1.5);
    view->GetInteractor()->Initialize();
    view->GetInteractor()->Start();
};
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

enum _layer_types {
    _neural_network_simple = 1,
    _convolutional_neural_network = 2
};

struct _layer {
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
    relu = 3,
    tanh = 4,
    tanh_sig = 5
};

_transport_func getTransportFunc(int);
namespace SDBP {    //Steepest Descent Backpropagation
    void __learn(MatrixXd, MatrixXd, double /* α */, vector<_layer>, int, double, function<VectorXd(VectorXd)> *);
}

namespace Batch_SDBP {    //Batch Steepest Descent Backpropagation
    void __learn(MatrixXd, MatrixXd, double /* α */, vector<_layer>, int, double, unsigned int, function<VectorXd(VectorXd)> *);
}

namespace CNN {
}

namespace VLBP {    //Variable Learning-rate Backpropagation
    void __learn(MatrixXd, MatrixXd, double /* mue */, double /* p */, double /* l */, vector<_layer>, int, double, function<VectorXd(VectorXd)> *);
}

namespace CGBP {    //Conjugate Gradient Backpropagation

}

namespace LMBP {    //Levenberg-Marquardt Backpropagation
    void __learn(MatrixXd, MatrixXd, double /* µ */,double /* ∂ */, vector<_layer>, int, double, function<VectorXd(VectorXd)> *);
}

namespace K_MEANS {
    void __learn(MatrixXd,unsigned int,unsigned int,function<unsigned int (VectorXd)>*);
}


#endif //BACKPROPAGATION_DEMO_LEARN_CPP_H
