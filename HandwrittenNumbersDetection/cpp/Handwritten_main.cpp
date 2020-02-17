#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <Eigen/Dense>
#include <iomanip>
#include <chrono>
#include <map>
#include <utility>
#include "../../AANN/Learn.h"
#include <fann.h>
#include <fann_cpp.h>

using namespace std;
using namespace Eigen;
using namespace chrono;

unsigned int reverseInt (int i);
void readIDXUByteFiles(char* images,char* labels,MatrixXd *input,MatrixXd *output,unsigned int w_cases);
void readIDXUByteFiles(char* images,char* labels,double **input,double **output,unsigned int w_cases);


void test01();
void test01FANN();
void test02();

int main(int argc,char** argv) {
    srand(time(NULL));

    test01();
//    test01FANN();
//    test02();

    return 0;
}


unsigned int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void readIDXUByteFiles(char* images,char* labels,MatrixXd *input,MatrixXd *output,unsigned int wanted_cases)
{

    ifstream labels_file(labels,ios::binary);
    ifstream images_file(images,ios::binary);

    if((!images_file.is_open()) || (!labels_file.is_open()))
    {
        cout << "[!] "  << "FAILED TO OPEN FILES"<< labels_file.is_open()<< images_file.is_open()  << endl;
        images_file.close();
        labels_file.close();
        return ;
    }

    unsigned int magic_number = 0,n_cases = 0,n_rows = 0 ,n_cols = 0;

    images_file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    images_file.read((char*)&n_cases, sizeof(n_cases));
    n_cases = reverseInt(n_cases);
    images_file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    images_file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);
    labels_file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    labels_file.read((char*)&n_cases, sizeof(n_cases));
    n_cases = reverseInt(n_cases);

    int dimentions = n_rows*n_cols;

    MatrixXd _in(dimentions,wanted_cases);
    MatrixXd _out (10,wanted_cases);

    const long double __NORMALIZE_NUM__ =  0.0078125;

    for(int i=0;i<wanted_cases;i++){
//        char reusable;
//        labels_file.read(&reusable, sizeof(reusable));
//        cases[i].label = reusable;

        VectorXd in(dimentions);
        VectorXd out(10);

        unsigned char ch =0;

        labels_file.read((char*)&ch, sizeof(unsigned char));

        for (int j = 0; j < 10; ++j) {
            out(j) = ((int)ch) == j;
        }

//        cout << (int) ch << " => "<< out.transpose() << endl;

        ch = 0;

        for(int r=0;r<dimentions;++r)
        {
            images_file.read((char*)&ch, sizeof(char));
            in(r) = ((double)ch)*__NORMALIZE_NUM__ -1;
        }

        _in.col(i) = in;
        _out.col(i) = out;
    }

    *input = _in;
    *output = _out;

    return;
}
typedef fann_type _data_type,*data_type_ptr;

void readIDXUByteFiles(char* images,char* labels,_data_type ***input,_data_type ***output,unsigned int wanted_cases) {

    ifstream labels_file(labels, ios::binary);
    ifstream images_file(images, ios::binary);

    if ((!images_file.is_open()) || (!labels_file.is_open())) {
        cout << "[!] " << "FAILED TO OPEN FILES" << labels_file.is_open() << images_file.is_open() << endl;
        images_file.close();
        labels_file.close();
        return;
    }

    unsigned int magic_number = 0, n_cases = 0, n_rows = 0, n_cols = 0;

    images_file.read((char *) &magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    images_file.read((char *) &n_cases, sizeof(n_cases));
    n_cases = reverseInt(n_cases);
    images_file.read((char *) &n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    images_file.read((char *) &n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);
    labels_file.read((char *) &magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    labels_file.read((char *) &n_cases, sizeof(n_cases));
    n_cases = reverseInt(n_cases);

    int dimentions = n_rows * n_cols;

    *input  = new _data_type*[wanted_cases];
    *output = new _data_type*[wanted_cases];

    const long double __NORMALIZE_NUM__ = 0.0078125;

    for (int i = 0; i < wanted_cases; i++) {
//        char reusable;
//        labels_file.read(&reusable, sizeof(reusable));
//        cases[i].label = reusable;

        (*input)[i]  = new _data_type[dimentions];
        (*output)[i] = new _data_type[10];

        unsigned char ch = 0;

        labels_file.read((char *) &ch, sizeof(unsigned char));

        for (int j = 0; j < 10; ++j) {
            (*output)[i][j] = ((int) ch) == j;
        }

//        cout << (int) ch << " => "<< out.transpose() << endl;

        ch = 0;

        for (int r = 0; r < dimentions; ++r) {
            images_file.read((char *) &ch, sizeof(char));
            (*input)[i][r] = ((double) ch) * __NORMALIZE_NUM__ - 1;
        }
    }

    return;
}

void test01 (){
    MatrixXd input, output;
    MatrixXd test_input, test_output;
    readIDXUByteFiles("/Users/atrinhojjat/Desktop/Numbers/train-images-idx3-ubyte",
                      "/Users/atrinhojjat/Desktop/Numbers/train-labels-idx1-ubyte", &input, &output,1);
    readIDXUByteFiles("/Users/atrinhojjat/Desktop/Numbers/t10k-images-idx3-ubyte",
                      "/Users/atrinhojjat/Desktop/Numbers/t10k-labels-idx1-ubyte", &test_input, &test_output,1);
    if (input.cols() == 0) {
        cout << "[!] " << "FAILED TO READ DATA" << endl;
        exit(-1);
    }

    cout << "[!] " << input.cols() << " Training Cases Loaded" << endl;
    cout << "[!] " << test_input.cols() << " Test Cases Loaded" << endl;

    const int max_iters = 1;
    const double learning_rate = 0.01;
    const int n_tests = 2;
    const double tolerance = 1e-2;

    cout << "[!] Starting ..." << endl;
    for (int i = 0; i < n_tests; ++i) {
        cout << "[!] Test " << i << " : " <<endl;

        vector<_layer> layers = vector<_layer>();

        layers.push_back(_layer(getTransportFunc(sigmoid), input.rows(), 2500));
        layers.push_back(_layer(getTransportFunc(sigmoid), 2500, 2000));
        layers.push_back(_layer(getTransportFunc(sigmoid), 2000, 1500));
        layers.push_back(_layer(getTransportFunc(sigmoid), 1500, 1000));
        layers.push_back(_layer(getTransportFunc(sigmoid), 1000, 500));
        layers.push_back(_layer(getTransportFunc(sigmoid), 500, 10));

//        layers.push_back(_layer(getTransportFunc(trans_funcs::tanh_sig), input.rows(), 1000));
//        layers.push_back(_layer(getTransportFunc(trans_funcs::tanh_sig), 1000, 500));
//        layers.push_back(_layer(getTransportFunc(trans_funcs::tanh_sig), 500, 10));

//        layers.push_back(_layer(getTransportFunc(trans_funcs::tanh_sig), input.rows(), 100));
//        layers.push_back(_layer(getTransportFunc(trans_funcs::tanh_sig), 100, 50));
//        layers.push_back(_layer(getTransportFunc(trans_funcs::tanh_sig), 50, 10));

        function<VectorXd(VectorXd)> iters_test;

        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        SDBP::__learn(input, output, learning_rate, layers, max_iters, tolerance, &iters_test);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(t2 - t1).count();

        cout << setw(6) << duration << " millisecs\n";
        double accuracy = 0;

        auto hardlim = [](VectorXd in) -> VectorXd {
            int max = 0;
//            cout << in << endl;
            for (int j = 0; j < in.size(); ++j) {
//                if(in(max) == in(j))
//                    cout << "tow answers" << in(max) << " " << in(j)  <<endl;
                if (in(max) < in(j)) {
                    max = j;
                }
            }
            VectorXd ret(in.size());
            for (int k = 0; k < in.size(); ++k) {
                ret(k) = k == max;
            }
            return ret;
        };

        for (int j = 0; j < test_input.cols(); ++j) {
            if (hardlim(iters_test(test_input.col(j))) == test_output.col(j))
                accuracy++;
        }

        accuracy /= test_input.cols();
        accuracy *=100;

        cout << "~~~ Test Results : " << accuracy << "%" << "out of" << test_input.cols() << endl ;

    }

    cout <<"[!] AI Training Done."<<endl
    << "[!] Exiting..."<<endl;

}

void test01FANN (){
    const unsigned int n_ins=60000, n_test_ins =10000;
    fann_type** input, **output;
    fann_type** test_input, **test_output;
    readIDXUByteFiles("/Users/atrinhojjat/Desktop/Numbers/train-images-idx3-ubyte",
                      "/Users/atrinhojjat/Desktop/Numbers/train-labels-idx1-ubyte", &input, &output,n_ins);
    readIDXUByteFiles("/Users/atrinhojjat/Desktop/Numbers/t10k-images-idx3-ubyte",
                      "/Users/atrinhojjat/Desktop/Numbers/t10k-labels-idx1-ubyte", &test_input, &test_output,
                      n_test_ins);

    FANN::training_data input_data = FANN::training_data();
    FANN::training_data test_data = FANN::training_data();

    input_data.set_train_data(n_ins,784,input,10,output);
    test_data.set_train_data(n_test_ins,784,test_input,10,test_output);

    cout << "[!] " << n_ins << " Training Cases Loaded" << endl;
    cout << "[!] " << n_test_ins << " Test Cases Loaded" << endl;

    const int max_iters = 1000;
    const double learning_rate = 0.01;
    const int n_tests = 5;
    const double tolerance = 1e-5;

    cout << "[!] Starting ..." << endl;
    for (int i = 0; i < n_tests; ++i) {
        cout << "[!] Test " << i << " : " <<endl;

        FANN::neural_net net = FANN::neural_net();

        const unsigned int  layers[] = {784,2500,2000,1500,1000,500,10};
        net.create_standard_array(7,&layers[0]);
//        const unsigned int  layers[] = {784,10};
//        net.create_standard_array(2,&layers[0]);

        net.set_learning_rate(learning_rate);

        net.set_training_algorithm(FANN::training_algorithm_enum::TRAIN_QUICKPROP);
        net.set_activation_function_hidden(FANN::activation_function_enum::SIGMOID);
        net.set_activation_function_output(FANN::activation_function_enum::SIGMOID);

        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        net.train_on_data(input_data,max_iters,100,tolerance);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(t2 - t1).count();

        cout << setw(6) << duration << " secs\n";
//        double accuracy = 0;

//        auto hardlim = [](_data_type* in,unsigned int dims) -> _data_type* {
//            int max = 0;
//            for (int j = 0; j < dims; ++j) {
//                if (in[max] < in[j]) {
//                    max = j;
//                }
//            }
//            _data_type *ret = new _data_type[dims];
//            for (int k = 0; k < dims; ++k) {
//                ret[k] = k == max;
//            }
//            return ret;
//        };

        cout << "Error on test set : " << net.test_data(test_data) << endl;

//        for (int j = 0; j < n_test_ins; ++j) {
//            if (hardlim(net.run(test_input[j]),10) == test_output[j])
//                accuracy++;
//        }
//
//        accuracy /= n_test_ins;
//        accuracy *=100;
//
//        cout << "~~~ Test Results : " << accuracy << "%" << "out of" << n_test_ins << endl ;

        string file_path = "/Users/atrinhojjat/Desktop/AI Results/Handwritten Numbers/Test";
        file_path+= (char)(i+48);
        cout << "Saving to " << file_path << endl;
        net.save(file_path);

    }

    cout <<"[!] AI Training Done."<<endl
    << "[!] Exiting..."<<endl;

}

void test02 () {
    //TODO: K-Means Clustering
    const unsigned int n_ins = 10000,n_test_ins = 10000;
    MatrixXd input, output;
    MatrixXd test_input, test_output;
    readIDXUByteFiles("/Users/atrinhojjat/Desktop/Numbers/train-images-idx3-ubyte",
                      "/Users/atrinhojjat/Desktop/Numbers/train-labels-idx1-ubyte", &input, &output,n_ins);
    readIDXUByteFiles("/Users/atrinhojjat/Desktop/Numbers/t10k-images-idx3-ubyte",
                      "/Users/atrinhojjat/Desktop/Numbers/t10k-labels-idx1-ubyte", &test_input, &test_output,n_test_ins);
    if (input.cols() == 0) {
        cout << "[!] " << "FAILED TO READ DATA" << endl;
        exit(-1);
    }

    cout << "[!] " << input.cols() << " Training Cases Loaded" << endl;
    cout << "[!] " << test_input.cols() << " Test Cases Loaded" << endl;

    const int max_iters = 1000;
    const double learning_rate = 0.01;
    const int n_tests = 2;
    const double tolerance = 1e-2;

    auto __correct_ans = [](VectorXd v)->int{
        for (int i = 0; i < v.rows(); ++i) {
            if(v(i) == 1) return i;
        }
    };

    cout << "[!] Starting ..." << endl;
    for (int i = 0; i < n_tests; ++i) {
        cout << "[!] Test " << i << " : " <<endl;

        function<unsigned int(VectorXd)> test_func;

        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        K_MEANS::__learn(input,10,1000,&test_func);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        auto duration = duration_cast<seconds>(t2 - t1).count();

        cout << setw(6) << duration << " secs\n";
        double accuracy = 0;

        typedef pair<unsigned int,unsigned int> CLSTR;
        map<CLSTR,unsigned int> clustering_results;
        map<unsigned int,unsigned int> res_map;

        for (int k = 0; k < 10; ++k) {
            for (int j = 0; j < 10; ++j) { //Over unlabeled clusters
                clustering_results[make_pair(j,k)] = 0;
            }
        }

        for (int j = 0; j <input.cols() ; ++j) {
            VectorXd _case = input.col(j);
            clustering_results[make_pair(test_func(_case),__correct_ans(output.col(j)))]++;
        }
        cout << endl << clustering_results.size() << endl;

        for (int k = 0; k < 10; ++k) {
            unsigned int res = 0;
            res = clustering_results[make_pair(k,0)];
            unsigned int best_res = 0;
            unsigned int best_val = res;
            for (int j = 1; j < 10; ++j) { //Over unlabeled clusters
                res = clustering_results[make_pair(k,j)];
                if(res > best_val){
                    best_res = j;
                    best_val = res;
                }
            }
            res_map[k] = best_res;
            cout << best_res << " -> " << k << endl;
        }

        for (int l = 0; l < test_input.cols(); ++l) {
            VectorXd v = test_input.col(l);
            if(res_map[test_func(v)] == __correct_ans(test_output.col(l))){
                accuracy++;
            }
        }

        accuracy /= test_input.cols();
        accuracy *=100;

        cout << "~~~ Test Results : " << accuracy << "%" << "out of" << test_input.cols() << endl ;

    }

    cout <<"[!] AI Training Done."<<endl
    << "[!] Exiting..."<<endl;

}
