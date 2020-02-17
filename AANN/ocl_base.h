//
// Created by Atrin Hojjat on 8/15/16.
//

#ifndef BACKPROPAGATION_DEMO_OCL_BASE_H
#define BACKPROPAGATION_DEMO_OCL_BASE_H

#include <functional>
#include <cmath>

using namespace std;

namespace AANN {
    namespace __base {
        struct __transport_function {
            __transport_function(function<float(float)> trans,function<float(float)> drans_drv) : transport(trans),transport_drv(trans_drv){};
            inline float transport(float);
            inline float transport_drv(float);
        };

#define __TRANS_FUNC__(name) inline float name(float in)
        namespace __trans_func_val {
            //TYPE 0 : PURE LINE

            __TRANS_FUNC__(pureline){
                return in;
            }
            __TRANS_FUNC__(pureline_drv){
                return 1;
            }

            //TYPE 1 : SIGMOID

            __TRANS_FUNC__(sigmoid){
                return 1/(exp(-in)+1);
            }
            __TRANS_FUNC__(sigmoid_drv){
                float sig = sigmoid(in);
                return sig*(1-sig);
            }

            //TYPE 2 : ReLU

            __TRANS_FUNC__(relu){
                return in>0 ? in : 0;
            }
            __TRANS_FUNC__(relu_drv){
                if(in == 0) return nan;
                return (float) in > 0;
            }

            //TYPE 3 : TANH

            __TRANS_FUNC__(tanh){
                return tanh(in);
            }
            __TRANS_FUNC__(tanh_drv){
                return 1-pow(tanh(in),2);
            }
        };
#undef __TRANS_FUNC__(name)

        struct __layer_struct {
            __layer_struct(float I,float O,float* w,float* b,auto transport,auto transport_derivative) :
                    I(I),O(O),w(w),b(b),transport(transport),transport_derivative(transport_derivative){

            };
            float* w,b;
            int I,O;
            float* outputs;
            int start_pnt;
            const __transport_function func;
        };

    };
    typedef __base::__layer_struct _layer;
    typedef __base::__transport_function _trans_func;
};

#endif //BACKPROPAGATION_DEMO_OCL_BASE_H
