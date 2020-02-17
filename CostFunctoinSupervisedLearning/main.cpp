//
// Created by Atrin Hojjat on 6/17/16.
//
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <fstream>
#include <sstream>
#include "Learn.h"
#include <chrono>


using namespace std;
using namespace std::chrono;
using namespace Eigen;

void test01(){
    double learning_rate = 1.745;
    double regularization_parameter = 0;

    MatrixXd trainingset(2,4),trainingoutput(1,4);

    auto createExample1D = [](double x)->double{
        return x+100;
    };

    for (int i = 0; i < 4; ++i) {
        trainingset(0,i) = 1;
        trainingset(1,i) = i*100;
        trainingoutput(0,i) = createExample1D(i*100);
    }

    function<VectorXd(VectorXd)> test;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    Learn ai = Learn(trainingset,trainingoutput,learning_rate,regularization_parameter);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<seconds>( t2 - t1 ).count();

    cout << "Runtime for " << ai.getIterationStats().size() << " Iterations : " << duration << " Seconds";


    test = ai.getTestFunc();

    cout << endl << endl << "Testing ...";

    srand(time(NULL));

    for (int j = 0; j < 30; ++j) {
        VectorXd intest(2);

        long random = rand()%10000;

        intest << 1,random;
        cout << endl << "Test for Input "<<random<<"(Expected Output = "<<createExample1D(random)<<") : " << test(intest);
    }
}

void test02(){
    double learning_rate = 0.17;
    double regularization_parameter = 0;

    ifstream input("test/data_test02.txt");
    if(!input.is_open())
        return;
    vector<vector<double>> data = vector<vector<double>>(47,vector<double>(3));
    int i=0,j = 0;
    double x;
    while(input>>x){
        data[i][j] = x;
        j = (j+1) % 3;
        if(j==0)
            i++;
    }

    input.close();

    for_each(data.begin(),data.end(),[](vector<double> x){cout << x[0] << ' ' << x[1] << ' ' << x[2] << endl; });

    MatrixXd inmat(3,47),outmat(1,47);

    for (int k = 0; k < 47; ++k) {
        inmat(0,k) = 1;
        for (int l = 0; l < 2; ++l) {
            inmat(l+1,k) = data[k][l];
        }
        outmat(0,k) = data[k][2];
    }

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    Learn ai = Learn(inmat,outmat,learning_rate,regularization_parameter);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<seconds>( t2 - t1 ).count();

    cout << "Runtime for " << ai.getIterationStats().size() << " Iterations : " << duration << " Seconds";
    function<VectorXd(VectorXd)> test = ai.getTestFunc();

}

void test03(){
/* TODO: Polynomial For Test 2 */
// MAX Number of Iterations is 100000
    double learning_rate = 0.17278;
    double regularization_parameter = 0;
    long max_iters = 10000000;

    ifstream input("test/data_test02.txt");
    if(!input.is_open())
        return;
    vector<vector<double>> data = vector<vector<double>>(47,vector<double>(3));
    int i=0,j = 0;
    double x;
    while(input>>x){
        data[i][j] = x;
        j = (j+1) % 3;
        if(j==0)
            i++;
    }

    input.close();

    for_each(data.begin(),data.end(),[](vector<double> x){cout << x[0] << ' ' << x[1] << ' ' << x[2] << endl; });

    MatrixXd inmat(6,47),outmat(1,47);

    for (int k = 0; k < 47; ++k) {
        inmat(0,k) = 1;
        for (int l = 0; l < 2; ++l) {
            inmat(l+1,k) = data[k][l];
        }
        inmat(3,k) = pow(data[k][0],2);
        inmat(4,k) = pow(data[k][1],2);
        inmat(5,k) = data[k][0]*data[k][1];
        outmat(0,k) = data[k][2];
    }

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    Learn ai = Learn(inmat,outmat,learning_rate,regularization_parameter,max_iters);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<seconds>( t2 - t1 ).count();

    cout << "Runtime for " << ai.getIterationStats().size() << " Iterations : " << duration << " Seconds";

    function<VectorXd(VectorXd)> test = ai.getTestFunc();

}
/* TODO Demo : use newton's method to find the extreme point of Performance function */
void test04(){
    ifstream input("test/data_test02.txt");
    if(!input.is_open())
        return;
    vector<vector<double>> data = vector<vector<double>>(47,vector<double>(3));
    int i=0,j = 0;
    double x;
    while(input>>x){
        data[i][j] = x;
        j = (j+1) % 3;
        if(j==0)
            i++;
    }

    input.close();

    for_each(data.begin(),data.end(),[](vector<double> x){cout << setw(4) << x[0] << ' ' << setw(1) << x[1] << ' ' << setw(6) << x[2] << endl; });

    MatrixXd inmat(3,47),outmat(1,47);

    for (int k = 0; k < 47; ++k) {
        inmat(0,k) = 1;
        for (int l = 0; l < 2; ++l) {
            inmat(l+1,k) = data[k][l];
        }
        outmat(0,k) = data[k][2];
    }

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    double c = 0;
    VectorXd d = VectorXd::Zero(3,1);
    MatrixXd A = MatrixXd::Zero(3,3);

    for (int m = 0;m<47;++m) {
        c+= outmat(0,m)*outmat(0,m);
        d = d + -2*outmat(0,m)*inmat.col(m);
        A = A + 2*inmat.col(m)*inmat.col(m).transpose();
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    VectorXd weighs=-A.inverse()*d;

    cout << weighs << endl <<endl;

    cout << c << endl <<endl;

    cout << d << endl <<endl;

    cout << A << endl <<endl;

    auto J = [A,d,c](VectorXd x)-> double{
        return (x.transpose() * A* x + x.transpose()*d)(0,0) +c;
    };

    auto J_norm = [inmat,outmat](MatrixXd w)-> double{
        double out = 0;

        for (int i = 0; i < inmat.cols(); ++i) {
            VectorXd delta = w.transpose()*inmat.col(i)-outmat.col(i);
            out+= delta.dot(delta);
        }

        out /= inmat.cols()*2;

        return out;
    };

    cout << "Performance Function : " << J(weighs) << "      " << J_norm(weighs);

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();

    cout << endl << "Time : " << duration << " Microseconds";

    /*
     * The Function has found the minimum
     * The Minimum of this function is 6.16561e+12 which is much higher than the result from Gradient Descent
     * but if we calculate the productivity use the function we used in gradient descent we will get the same value
     * as gradient descent in only one iterations.
     * if matrix A isn't too big and the determinant is not equal to zero then this method will work much better
     * than gradient descent.
     *
     * Note: This is only functional for Quadratic functions.for other functions it has an unpredicted result based
     *       on starting point
     * */


}

int main(int argc,char** argv)
{

//    test01();
//    test02();
//    test03();
    test04(); //Test Newton's Method

    return 0;
}
