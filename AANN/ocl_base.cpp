//
// Created by Atrin Hojjat on 8/15/16.
//

#include "ocl_base.h"

using namespace AANN;

const _trans_func
        sigmoid (&__base::__trans_func_val::sigmoid ,&__base::__trans_func_val::sigmoid_drv ),
        pureline(&__base::__trans_func_val::pureline,&__base::__trans_func_val::pureline_drv),
        relu    (&__base::__trans_func_val::relu    ,&__base::__trans_func_val::relu_drv    ),
        tanh    (&__base::__trans_func_val::tanh    ,&__base::__trans_func_val::tanh_drv    );

