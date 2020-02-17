//
// Created by Atrin Hojjat on 6/17/16.
//

#ifndef PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_MATRIX_H
#define PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_MATRIX_H

#include <cmath>
#include <iomanip>
#include <vector>

#define CALL(func, args...) do {if (func) (func)(## args);} while (0)

using namespace std;

struct _matrix {
    _matrix(int i,int j) : i(i),j(j){
        mat = vector<vector<double>>(i,vector<double>(j));
    };
    const int i;
    const int j;
    vector<vector<double>> mat;

};

void _matrix_print(_matrix mat);
_matrix _matrix_O(int i,int j);
_matrix _matrix_I(int i);
_matrix _matrix_powT(_matrix mat);
_matrix _matrix_multiply(_matrix mat1,_matrix mat2,int *error);
double _matrix_determinant(_matrix mat,int *error);
void _matrix_print(_matrix mat);

_matrix _matrix_O(int i,int j){
    _matrix ret(i,j);
    fill(&ret.mat[0][0],&ret.mat[i-1][j-1],0);
    return ret;
}

_matrix _matrix_I(int i){
    _matrix ret(i,i);
    fill(&ret.mat[0][0],&ret.mat[i-1][i-1],0);

    for(int k=0;k<i;k++){
        ret.mat[k][k]=1;
    }
    return ret;
}

_matrix _matrix_powT(_matrix mat){
    _matrix ret(mat.j,mat.i);
    for(int j=0;j<mat.j;j++){
        for(int i=0;i<mat.i;i++){
            ret.mat[j][i] = mat.mat[i][j];
        }
    }
    return ret;
}

_matrix _matrix_multiply(_matrix mat1,_matrix mat2,int *error){
    if(mat1.j!=mat2.i){
        *(error) = -1;
        return _matrix_O(mat1.i,mat2.j);
    }
    _matrix ret(mat1.i,mat2.j);

    auto inner_product = [](double* x,double* y,int count) {
        double p = 0;
        for (int i = 0; i < count; i++) {
            p += x[i] * y[i];
        }
        return p;

    };

    for (int i = 0; i < mat1.i; ++i) {
        for (int j = 0; j < mat2.j; ++j) {
            ret.mat[i][j] = inner_product(&mat1.mat[i][0],&_matrix_powT(mat2).mat[i][0],mat1.j);
        }
    }
    return ret;
}

double _matrix_determinant(_matrix mat,int *error)
{
    if(mat.i!=mat.j){
        *(error) = -1;
        return 0;
    }else {
        if(mat.i == 1){
            return mat.mat[0][0];
        }else{
            auto _matrix_split = [](_matrix mat,int col){ /*Simplified : Removes First Row of Matrix and One Column*/
                _matrix ret(mat.i-1,mat.j-1);
                for (int i = 1,newi=0; i < mat.i; ++i,newi++) {
                    for (int j = 0,newj=0; j < mat.j; ++j,newj++) {
                        if(j==col){
                            newj--;
                            continue;
                        }
                        ret.mat[newi][newj]=mat.mat[i][j];
                    }
                }

                return ret;
            };
            double d = 0;
            for (int i = 0; i < mat.i; ++i) {
                d+=pow(-1,i)*(mat.mat[0][i])*_matrix_determinant(_matrix_split(mat,i),error);
            }
            *(error) = 0;
            return d;
        }
    }
}

void _matrix_print(_matrix mat){
    for (int i = 0; i < mat.i; ++i) {
        for (int j = 0; j < mat.j; ++j) {
            cout << setw(5) << mat.mat[i][j] ;
        }
        cout << endl;
    }
}

#endif //PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_MATRIX_H
