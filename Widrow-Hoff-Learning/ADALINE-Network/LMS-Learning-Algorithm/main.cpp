//
// Created by Atrin Hojjat on 7/13/16.
//

#include <iostream>
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <fstream>
#include <sstream>
#include "Learn.h"
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;
using namespace Eigen;


/* TODO : Recognize T,G & F (Pattern Recognition)*/
void test01 () {
    MatrixXd inmat = MatrixXd::Zero(17,6);
    MatrixXd inmat_temp = MatrixXd::Zero(6,17);
    MatrixXd outmat = MatrixXd::Zero(1,6);

    outmat << 60 ,0,-60,60 ,0,-60;

    inmat_temp << 
             1,-1,-1,-1,  1, 1, 1, 1,  1,-1,-1,-1, -1,-1,-1,-1,  1,
             1, 1, 1, 1,  1,-1, 1, 1,  1,-1, 1, 1, -1,-1,-1,-1,  1,
             1, 1, 1, 1,  1, 1,-1,-1,  1,-1,-1,-1, -1,-1,-1,-1,  1,
            -1,-1,-1,-1,  1,-1,-1,-1,  1, 1, 1, 1,  1,-1,-1,-1,  1,
            -1,-1,-1,-1,  1, 1, 1, 1,  1,-1, 1, 1,  1,-1, 1, 1,  1,
             1, 1, 1, 1,  1, 1,-1,-1,  1,-1,-1,-1, -1,-1,-1,-1,  1;
    inmat = inmat_temp.transpose();

    for(int i=0;i<inmat.cols();i++){
        for (int j = 0; j < sqrt(inmat.rows()-1); ++j) {
            cout << (inmat(j,i)>0) << ' ' << (inmat(j+4,i)>0) << ' ' << (inmat(j+8,i)>0) << ' ' << (inmat(j+12,i)>0) << ' '<<endl;
        }
        cout <<endl;
    }


    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    Learn ai = Learn(inmat,outmat,0.03,0,10);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<seconds>( t2 - t1 ).count();

    cout << "Runtime for " << ai.getIterationStats().size() << " Iterations : " << duration << " Seconds";

    auto test = ai.getTestFunc();

    cout <<endl<< endl;

    for(int i=0;i<inmat.cols();i++){
        for (int j = 0; j < sqrt(inmat.rows()-1); ++j) {
            cout << (inmat(j,i)>0) << ' ' << (inmat(j+4,i)>0) << ' ' << (inmat(j+8,i)>0) << ' ' << (inmat(j+12,i)>0) << ' '<<endl;
        }
        cout << "="<< test(inmat.col(i)) << endl<<endl;
    }
    cout <<endl <<endl;

}

int main(int argc,char** argv){
    test01();
    return 0;
}