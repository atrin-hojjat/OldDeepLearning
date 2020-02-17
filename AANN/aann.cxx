#include "ocl_base.h"


//Feed Forward
__kernel void neuron (__global _layer* layers,__global int total_neurons,__global int input_dim,_global float* output){
    int input_num = get_global_id(0);
    int layer_num = get_global_id(1);
    int neuron_num = get_global_id(2);

    _layer l = layers[layer_num];

    int in_loc = input_num*total_neurons+(layer_num ==0 ? 0: layers[layer_num-1].start_pnt);
    int out_loc = in_loc+(layer_num ==0 ? input_dim: layers[layer_num-1].O);

    float out = 0;

    for(int i=0;i<l.I;i++){
        out+=l.w[l.O+i]*output[in_loc+i];
    }

    out += l.b[neuron_num];

    output[out_loc+neuron_num] = l.func.transport(out);
}

//Backprapogation
//Phase 1 : Feed forward