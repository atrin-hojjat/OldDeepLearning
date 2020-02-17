//
// Created by Atrin Hojjat on 8/5/16.
//

#ifndef BACKPROPAGATION_DEMO_KERNEL_H
#define BACKPROPAGATION_DEMO_KERNEL_H

#include <string>
#include <vector>
#include <algorithm>
#include <OpenCL/opencl.h>

using namespace std;

namespace OCL {
    typedef string              _std_str,           *_std_str_ptr;
    typedef int                 _iteration_type,    *_iteration_type_ptr;
    typedef int                 _dimension_num,     *_dimension_num_ptr;
    typedef const char *        _c_std_str,         *_c_std_str_ptr;
    typedef cl_device_id        _device_id,         *_device_id_ptr;
    typedef cl_context          _context,           *_context_ptr;
    typedef cl_command_queue    _command_queue,     *_command_queue_ptr;
    typedef cl_program          _program,           *_program_ptr;
    typedef cl_kernel           _kernel,            *_kernel_ptr;
    typedef cl_mem              _memory,            *_memory_ptr;

    typedef struct {
        _device_id device_id;             // compute device id
        _context context;                 // compute context
        _command_queue commands;          // compute command queue
        _program program;                 // compute program
        _kernel kernel;                   // compute kernel
    } __ocl,*__ocl_ptr;

    int init(_ocl_ptr,_std_str);

    template <_iteration_num I,_ocl_ptr oclPtr>
            void _create_memory();
    template <_iteration_num I,_ocl_ptr oclPtr,typename type01,typename ..._types>
            void _create_memory(type01 arg01,_types...args);
    template <_c_std_str name,_ocl_ptr oclPtr,typename ..._types>
            void _call(_dimension_num dim_num,size_t* global_work_size,size_t* local_work_size,_types...args);
}

#endif //BACKPROPAGATION_DEMO_KERNEL_H
