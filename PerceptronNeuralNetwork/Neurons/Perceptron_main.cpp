//
// Created by Atrin Hojjat on 5/30/16.
//

#include <iostream>
#include "DEFINITIONS.h"
#include "PerceptronNetwork.h"

using namespace std;

void listener(){cout << "listener";}

int main(int argc,char** argv){
    vector<__case> input(8);

    int pos[8][2] =    {{1,1},{1,2},{2,-1},{2,0},{-1,2},{-2,1},{-1,-1},{-2,-2}};
    bool out[8][2] =   {{0,0},{0,0},{0, 1},{0,1},{ 1,0},{ 1,0},{ 1, 1},{ 1, 1}};

    for(int i=0;i<8;i++){
        input[i].output = *new vector<bool>(2);
        input[i].input = *new vector<double>(2);
        input[i].output[0]=out[i][0];
        input[i].output[1]=out[i][1];
        input[i].input[0] = pos[i][0];
        input[i].input[1] = pos[i][1];

    }

    PerceptronNetwork *net = new PerceptronNetwork(input,2,&listener);
}