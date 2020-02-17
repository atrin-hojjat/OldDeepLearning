//
// Created by Atrin Hojjat on 6/17/16.
//

#include "DEFINITIONS.h"
#include "SupervisedHebbianNetwork.h"
#include "Learn.h"

int training_number=3,input_dimension_number=30,output_dimension_number=30;

MatrixXd mat_hardlim(MatrixXd mat){
    MatrixXd ret(mat.rows(),mat.cols());
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            ret(i,j) = hardlim(mat(i,j));
        }
    }
    return ret;
}
MatrixXd mat_hardlims(MatrixXd mat){
    MatrixXd ret(mat.rows(),mat.cols());
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            ret(i,j) = hardlims(mat(i,j));
        }
    }
    return ret;
}

void test_50_percent_removal(MatrixXd W){
    VectorXd p0(input_dimension_number);
    VectorXd p1(input_dimension_number);
    VectorXd p2(input_dimension_number);

    p0 <<
            -1, 1, 1, 1,-1,
             1,-1,-1,-1, 1,
             1,-1,-1,-1, 1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1;
    p1<<
            -1, 1, 1,-1,-1,
            -1,-1, 1,-1,-1,
            -1,-1, 1,-1,-1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1;
    p2<<
             1, 1, 1,-1,-1,
            -1,-1,-1, 1,-1,
            -1,-1,-1, 1,-1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1;
    cout <<endl << "0 =>"<<mat_hardlim(W*p0).transpose();
    cout <<endl << "1 =>"<<mat_hardlim(W*p1).transpose();
    cout <<endl << "2 =>"<<mat_hardlim(W*p2).transpose();

}
void test_66_percent_removal(MatrixXd W){
    VectorXd p0(input_dimension_number);
    VectorXd p1(input_dimension_number);
    VectorXd p2(input_dimension_number);

    p0 <<
            -1, 1, 1, 1,-1,
             1,-1,-1,-1, 1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1;
    p1<<
            -1, 1, 1,-1,-1,
            -1,-1, 1,-1,-1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1;
    p2<<
             1, 1, 1,-1,-1,
            -1,-1,-1, 1,-1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1;
    cout <<endl << "0 =>"<<mat_hardlim(W*p0).transpose();
    cout <<endl << "1 =>"<<mat_hardlim(W*p1).transpose();
    cout <<endl << "2 =>"<<mat_hardlim(W*p2).transpose();
}
void test_noise(MatrixXd W){
    VectorXd p0(input_dimension_number);
    VectorXd p1(input_dimension_number);
    VectorXd p2(input_dimension_number);

    p0 <<
            -1,-1, 1, 1,-1,
             1,-1,-1, 1,-1,
             1,-1,-1,-1, 1,
            -1,-1,-1,-1,-1,
             1,-1, 1,-1, 1,
            -1, 1,-1, 1,-1;
    p1<<
            -1, 1, 1,-1,-1,
            -1,-1, 1, 1,-1,
             1,-1, 1,-1,-1,
            -1,-1,-1,-1, 1,
            -1,-1, 1,-1,-1,
             1, 1, 1, 1,-1;
    p2<<
            -1, 1, 1,-1,-1,
            -1, 1,-1, 1,-1,
            -1,-1,-1,-1,-1,
            -1, 1, 1,-1, 1,
            -1,-1,-1,-1,-1,
             1, 1, 1,-1, 1;
    cout <<endl << "0 =>"<<mat_hardlim(W*p0).transpose();
    cout <<endl << "1 =>"<<mat_hardlim(W*p1).transpose();
    cout <<endl << "2 =>"<<mat_hardlim(W*p2).transpose();
}


int main(int argc,char** argv)
{
    vector<VectorXd> p(training_number);
    vector<VectorXd> t(training_number);

    MatrixXd W;
    for (int j = 0; j < training_number; ++j) {
        p[j] = VectorXd(input_dimension_number);
        t[j] = VectorXd(output_dimension_number);
    }

    p[0] <<
            -1, 1, 1, 1,-1,
             1,-1,-1,-1, 1,
             1,-1,-1,-1, 1,
             1,-1,-1,-1, 1,
             1,-1,-1,-1, 1,
            -1, 1, 1, 1,-1;
    p[1]<<
            -1, 1, 1,-1,-1,
            -1,-1, 1,-1,-1,
            -1,-1, 1,-1,-1,
            -1,-1, 1,-1,-1,
            -1,-1, 1,-1,-1,
            -1,-1, 1,-1,-1;
    p[2]<<
             1, 1, 1,-1,-1,
            -1,-1,-1, 1,-1,
            -1,-1,-1, 1,-1,
            -1, 1, 1,-1,-1,
            -1, 1,-1,-1,-1,
            -1, 1, 1, 1,-1;
    t[0] <<
            -1, 1, 1, 1,-1,
             1,-1,-1,-1, 1,
             1,-1,-1,-1, 1,
             1,-1,-1,-1, 1,
             1,-1,-1,-1, 1,
            -1, 1, 1, 1,-1;
    t[1]<<
            -1, 1, 1,-1,-1,
            -1,-1, 1,-1,-1,
            -1,-1, 1,-1,-1,
            -1,-1, 1,-1,-1,
            -1,-1, 1,-1,-1,
            -1,-1, 1,-1,-1;
    t[2]<<
             1, 1, 1,-1,-1,
            -1,-1,-1, 1,-1,
            -1,-1,-1, 1,-1,
            -1, 1, 1,-1,-1,
            -1, 1,-1,-1,-1,
            -1, 1, 1, 1,-1;

    vector<__case> study_cases(training_number);

    for (int i = 0; i < training_number; ++i) {
        study_cases[i] = __case();
        study_cases[i].input = p[i];
        study_cases[i].output = p[i];
    }

    vector<vector<MatrixXd>> *weights = new vector<vector<MatrixXd>>(1,vector<MatrixXd>(1));

    auto listener = [](){};

    vector<int>neurons_num = vector<int>(1);
    neurons_num[0] = 1;

    SupervisedHebbianNetwork network = SupervisedHebbianNetwork(&study_cases,1,neurons_num,listener);

    W = *(network.getWeight()[0][0]);

    cout << "Weight Matrix Generated : "<< endl << W
        <<endl << "Start Testing";

    cout <<endl << "0 =>"<<mat_hardlim(W*p[0]).transpose();
    cout <<endl << "1 =>"<<mat_hardlim(W*p[1]).transpose();
    cout <<endl << "2 =>"<<mat_hardlim(W*p[2]).transpose();

    cout << "\n50% Removal :";
    test_50_percent_removal(W);
    cout << "\n66% Removal :";
    test_66_percent_removal(W);
    cout << "\nWith Niose :";
    test_noise(W);


    return 0;
}