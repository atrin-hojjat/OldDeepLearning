//
// Created by Atrin Hojjat on 7/20/16.
//

#include <iostream>
#include <functional>
#include <vector>
#include <Eigen/Dense>
#include <math.h>
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
#include <chrono>
#include <vtkPen.h>
#include <fann.h>
#include <floatfann.h>

#include "../../AANN/Learn.h"

using namespace std;
using namespace std::chrono;
using namespace Eigen;

void test01SDBP(){
    const int input_num = 50;
    const int test_num = 6;
    const int point_num = 1000;
    const int max_iters = 3000;
    const double learning_rate = 0.125;
    const double noise = 0.01;

    auto goal_func = [](double x)-> double{
      return log ( x + 3 ) / log ( 5 ) * sin ( M_PI * x );
//        return abs ( sin ( M_PI * x /2 ) );
//        return cos(3*M_PI*x/2);
    };

    MatrixXd input(1,input_num),output(1,input_num);

    vector<function<VectorXd(VectorXd)>> tests = vector<function<VectorXd(VectorXd)>>(10);

    vtkSmartPointer<vtkTable> table =
            vtkSmartPointer<vtkTable>::New();

    vtkSmartPointer<vtkFloatArray> arrX =
            vtkSmartPointer<vtkFloatArray>::New();
    arrX->SetName("X");
    table->AddColumn(arrX);

    vtkSmartPointer<vtkFloatArray> arrY =
            vtkSmartPointer<vtkFloatArray>::New();
    arrY->SetName("Y - main");
    table->AddColumn(arrY);

    vtkSmartPointer<vtkFloatArray> arrC[test_num];
    for (int j = 0; j < test_num; ++j) {
        arrC[j] =
                vtkSmartPointer<vtkFloatArray>::New();

        arrC[j]->SetName(("Test "+ to_string( j)).c_str());
        table->AddColumn(arrC[j]);
    }

    table->SetNumberOfRows(point_num);

    double in_dis = 4*1.0/input_num;
    for (int i = 0; i < input_num; ++i) {
        double k = -2 + in_dis*i;
        input(0,i) = k;
        output(0,i) = goal_func(k) + noise*(rand()%100)*1.0/1000;
    }

    double distance = 4*1.0/point_num;
//    cout << point_num;
    for (int j = 0; j < point_num; ++j) {
        table->SetValue(j, 0, -2+j*distance);
        table->SetValue(j, 1, goal_func(-2+j*distance));
    }

    cout << endl;

    long double mean_err = 0.0;

    for (int j = 0; j < test_num; ++j) {
        cout << "Test " << setw(3) << j+1 << " -> ";
        vector<_layer> layers = vector<_layer>();

        /*
        layers.push_back(_layer(getTransportFunc(trans_funcs::sigmoid),1,5));
        layers.push_back(_layer(getTransportFunc(trans_funcs::relu),5,7));
        layers.push_back(_layer(getTransportFunc(trans_funcs::sigmoid),7,8));
        layers.push_back(_layer(getTransportFunc(trans_funcs::pureline),8,1));
         */
        layers.push_back(_layer(getTransportFunc(trans_funcs::sigmoid),1,7));
        layers.push_back(_layer(getTransportFunc(trans_funcs::sigmoid),7,7));
        layers.push_back(_layer(getTransportFunc(trans_funcs::sigmoid),7,7));
        layers.push_back(_layer(getTransportFunc(trans_funcs::pureline),7,1));

        function<VectorXd(VectorXd)> iters_test;

        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        SDBP::__learn(input,output,learning_rate,layers,max_iters,1e-5,&iters_test);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        auto duration = duration_cast<seconds>( t2 - t1 ).count();

        cout << setw(3) << duration << " secs\n";

        tests[j] = iters_test;

        double err = 0;
        double x ;
        double out;
        for (int i = 0; i < point_num; ++i) {
            x =-2 + i *distance;
            VectorXd v(1);
            v<< x;
            out = iters_test(v)(0);
            table->SetValue(i, 2+ j, out);
            err += pow(goal_func(x)-out,2);
//            cout << v << "               "<< iters_test(v) << endl;
        }
        err/= point_num;
        mean_err+= err;
        cout << "Mean Squered Error : " << err << endl;
    }

    mean_err/=test_num;

    cout << endl << endl << " -> Mean Error : " << mean_err << endl;

    // Set up the view
    vtkSmartPointer<vtkContextView> view =
            vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);

    // Add multiple line plots, setting the colors etc
    vtkSmartPointer<vtkChartXY> chart =
            vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    vtkPlot *line = chart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, 1);
    line->SetColor(0, 0, 0);
    line->SetWidth(1.5);
    for (int k = 0; k < test_num; ++k) {
        line = chart->AddPlot(vtkChart::LINE);
        double random,r,g,b;
        random = rand();
        r = random/(pow(10,(int)log10(random)+1)) ;
        random = rand();
        g = random/(pow(10,(int)log10(random)+1)) ;
        random = rand();
        b = random/(pow(10,(int)log10(random)+1)) ;
        line->SetInputData(table, 0, 2+k);
        line->SetColor(r, g, b);
        line->SetWidth(1.5);
    }
    view->GetInteractor()->Initialize();
    view->GetInteractor()->Start();
}

void test01LMBP(){
    const int input_num = 40;
    const int test_num = 10;
    const int point_num = 80;
    const int max_iters = 1000;
    const double mue = 0.001;
    const double delta = 2;
    const double noise = 1;

    auto goal_func = [](double x)-> double{
        return sin(M_PI*x/2);
//        return cos(3*M_PI*x/2);
    };

    MatrixXd input(1,input_num),output(1,input_num);

    vector<function<VectorXd(VectorXd)>> tests = vector<function<VectorXd(VectorXd)>>(10);

    vtkSmartPointer<vtkTable> table =
            vtkSmartPointer<vtkTable>::New();

    vtkSmartPointer<vtkFloatArray> arrX =
            vtkSmartPointer<vtkFloatArray>::New();
    arrX->SetName("X");
    table->AddColumn(arrX);

    vtkSmartPointer<vtkFloatArray> arrY =
            vtkSmartPointer<vtkFloatArray>::New();
    arrY->SetName("Y - main");
    table->AddColumn(arrY);

    vtkSmartPointer<vtkFloatArray> arrC[test_num];
    for (int j = 0; j < test_num; ++j) {
        arrC[j] =
                vtkSmartPointer<vtkFloatArray>::New();

        arrC[j]->SetName(("Test "+ to_string( j)).c_str());
        table->AddColumn(arrC[j]);
    }

    table->SetNumberOfRows(point_num);

    double in_dis = 0.1;
    for (int i = 0; i < input_num; ++i) {
        double k = -2 + in_dis*i;
        input(0,i) = k;
        output(0,i) = goal_func(k) + noise*(rand()%100)*1.0/1000;
    }

    double distance = 0.05;
//    cout << point_num;
    for (int j = 0; j < point_num; ++j) {
        table->SetValue(j, 0, -2+j*distance);
        table->SetValue(j, 1, goal_func(-2+j*distance));
    }

    cout << endl;

    long double mean_err = 0.0;

    for (int j = 0; j < test_num; ++j) {
        cout << "Test " << setw(3) << j+1 << " -> ";
        vector<_layer> layers = vector<_layer>();

        layers.push_back(_layer(getTransportFunc(trans_funcs::sigmoid),1,32));
        layers.push_back(_layer(getTransportFunc(trans_funcs::pureline),32,1));

        function<VectorXd(VectorXd)> iters_test;

        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        LMBP::__learn(input,output,mue,delta,layers,max_iters,1e-5,&iters_test);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        auto duration = duration_cast<seconds>( t2 - t1 ).count();

        cout << setw(3) << duration << " secs\n";

        tests[j] = iters_test;

        double err = 0;
        double x ;
        double out;
        for (int i = 0; i < point_num; ++i) {
            x =-2 + i *distance;
            VectorXd v(1);
            v<< x;
            out = iters_test(v)(0);
            table->SetValue(i, 2+ j, out);
            err += pow(goal_func(x)-out,2);
//            cout << v << "               "<< iters_test(v) << endl;
        }
        err/= point_num;
        mean_err+= err;
        cout << "Mean Squered Error : " << err << endl;
    }

    mean_err/=test_num;

    cout << endl << endl << " -> Mean Error : " << mean_err << endl;

    // Set up the view
    vtkSmartPointer<vtkContextView> view =
            vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);

    // Add multiple line plots, setting the colors etc
    vtkSmartPointer<vtkChartXY> chart =
            vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    vtkPlot *line = chart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, 1);
    line->SetColor(0, 0, 0);
    line->SetWidth(1.5);
    for (int k = 0; k < test_num; ++k) {
        line = chart->AddPlot(vtkChart::LINE);
        double random,r,g,b;
        random = rand();
        r = random/(pow(10,(int)log10(random)+1)) ;
        random = rand();
        g = random/(pow(10,(int)log10(random)+1)) ;
        random = rand();
        b = random/(pow(10,(int)log10(random)+1)) ;
        line->SetInputData(table, 0, 2+k);
        line->SetColor(r, g, b);
        line->SetWidth(1.5);
    }
    view->GetInteractor()->Initialize();
    view->GetInteractor()->Start();
}

void test01FANN(){
    const int input_num = 40;
    const int test_num = 10;
    const int point_num = 80;
    const int max_iters = 1000;
    const double learning_rate = 0.1;
    const double noise = 1;

    auto goal_func = [](double x)-> double{
        return sin(M_PI*x/2);
//        return cos(3*M_PI*x/2);
    };

    vtkSmartPointer<vtkTable> table =
            vtkSmartPointer<vtkTable>::New();

    vtkSmartPointer<vtkFloatArray> arrX =
            vtkSmartPointer<vtkFloatArray>::New();
    arrX->SetName("X");
    table->AddColumn(arrX);

    vtkSmartPointer<vtkFloatArray> arrY =
            vtkSmartPointer<vtkFloatArray>::New();
    arrY->SetName("Y - main");
    table->AddColumn(arrY);

    vtkSmartPointer<vtkFloatArray> arrC[test_num];
    for (int j = 0; j < test_num; ++j) {
        arrC[j] =
                vtkSmartPointer<vtkFloatArray>::New();

        arrC[j]->SetName(("Test "+ to_string( j+1)).c_str());
        table->AddColumn(arrC[j]);
    }

    table->SetNumberOfRows(point_num);

    {
        float x,y;
        for (int j = 0; j < 80; ++j) {
            x = -2+j*0.05;y=goal_func(x);

            table->SetValue(j, 0, x);
            table->SetValue(j, 1, y);
        }


        for (int t = 0; t < test_num; ++t) {
            x=0;y=0;
            cout << "[!] Test " << t+1 << " : " << endl;
            fann_type **input  = new fann_type*[input_num];
            fann_type **output = new fann_type*[input_num];

            for (int j = 0; j < 40; ++j) {
                x = -2+j*0.1;y=goal_func(x);
                input[j]=new float[1];
                output[j]=new float[1];
                input[j][0] = x;
                output[j][0] = y+ noise*(rand()%100)*1.0/100;
            }

            const unsigned int num_layers = 3;
            const unsigned int layers[num_layers] = {1,32,1};
            struct fann *ann = fann_create_standard_array(num_layers,layers);

            fann_set_activation_function_hidden(ann,FANN_SIGMOID);
            fann_set_activation_function_output(ann,FANN_LINEAR);
            fann_set_learning_rate(ann,learning_rate);
            fann_set_training_algorithm(ann,fann_train_enum::FANN_TRAIN_INCREMENTAL);

            struct fann_train_data *train_data = fann_create_train(input_num,1,1);
            train_data->input = input;train_data->output = output;


            high_resolution_clock::time_point t1 = high_resolution_clock::now();
            fann_train_on_data(ann,train_data,max_iters,10000,1e-5);
            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            auto duration = duration_cast<seconds>( t2 - t1 ).count();

            cout << "~~~ In " << duration << " Seconds" << endl;

            float err = 0;
            for (int j = 0; j < point_num; ++j) {
                float x = -2 + j*0.05;
                float out = *fann_run(ann,&x);

                table->SetValue(j, 2+ t, out);
                err += pow(goal_func(x)-out,2);
            }
            err/=point_num;
            cout << "~~~ Mean Squered Error : " << err << endl;
        }

    }

    {
        // Set up the view
        vtkSmartPointer<vtkContextView> view =
                vtkSmartPointer<vtkContextView>::New();
        view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);

        // Add multiple line plots, setting the colors etc
        vtkSmartPointer<vtkChartXY> chart =
                vtkSmartPointer<vtkChartXY>::New();
        view->GetScene()->AddItem(chart);
        vtkPlot *line = chart->AddPlot(vtkChart::LINE);
        line->SetInputData(table, 0, 1);
        line->SetColor(0, 0, 0);
        line->SetWidth(1.5);
        for (int k = 0; k < test_num; ++k) {
            line = chart->AddPlot(vtkChart::LINE);
            double random,r,g,b;
            random = rand();
            r = random/(pow(10,(int)log10(random)+1)) ;
            random = rand();
            g = random/(pow(10,(int)log10(random)+1)) ;
            random = rand();
            b = random/(pow(10,(int)log10(random)+1)) ;
            line->SetInputData(table, 0, 2+k);
            line->SetColor(r, g, b);
            line->SetWidth(1.5);
        }
        view->GetInteractor()->Initialize();
        view->GetInteractor()->Start();
    }
}

void LMBPtest(){
    vector<_layer> layers = vector<_layer>();

    layers.push_back(_layer(_transport_func(
            [](VectorXd in)->VectorXd{
                VectorXd ret(in.size());
                for (int i = 0; i < in.size(); ++i) {
                    ret(i) = pow(in(i),2);
                }
                return ret;
            },[](VectorXd in)->VectorXd {
                VectorXd ret(in.size());
                for (int i = 0; i < in.size(); ++i) {
                    ret(i) = in(i)* 2;
                }
                return ret;
            }), 1,1));
    layers.push_back(_layer(getTransportFunc(pureline),1,1));

    {
        MatrixXd w1(1,1);
        w1<<1;
        layers[0].weighs =w1;
        VectorXd b1(1);
        b1<<0;
        layers[0].bias = b1;
    }
    {
        MatrixXd w1(1,1);
        w1<<2;
        layers[1].weighs =w1;
        VectorXd b1(1);
        b1<<1;
        layers[1].bias = b1;
    }

    function<VectorXd(VectorXd)> iters_test;

    MatrixXd in(1,2),out(1,2);

    in<<1,2;
    out<<1,2;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    LMBP::__learn(in,out,0.1,10,layers,1,1,&iters_test);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<seconds>( t2 - t1 ).count();
}

int main(int argc,char** argv){
    srand(time(NULL));  //Should be Called only one time
   // LMBPtest();
      test01SDBP();
 //     test01FANN();
//    test01LMBP();
    return 0;
}