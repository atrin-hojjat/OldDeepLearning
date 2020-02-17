//
// Created by Atrin Hojjat on 8/5/16.
//

#include "kernel.h"

int OCL::init(__ocl_ptr ptr,__std_str prog_file){

    // Connect to a compute device
    //
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &(ptr->device_id), NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    // Create a compute context
    //
    ptr->context = clCreateContext(0, 1, &(ptr->device_id), NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    ptr->commands = clCreateCommandQueue(context, ptr->device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    ifstream infile(prog_file);
    if(!infile.is_open())
    {
        printf("Error: Failed to Read OCL File!\n");
        return EXIT_FAILURE;
    }
    string file("");
    string line;
    while(std::getline(infile, line)){
        file += line +'\n';
    }
    file += '\0';
    __c_std_str_ptr file_ptr = file.c_str();

    // Create the compute program from the source buffer
    //
    ptr->program = clCreateProgramWithSource(ptr->context, 1, file_ptr, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(ptr->program, 1, &(ptr->device_id), "", NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(ptr->program, ptr->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
}

void OCL::_create_memory() {
    cout << I << " Parameters Were set successfully.";
}

void OCL::_create_memory(type01 arg01, _types ...args) {
    clSetKernelArg(oclPtr->kernel, I, sizeof(cl_mem), &arg01);
    _create_memory<I+1,oclPtr>(args...);
}

void OCL::_call(_dimension_num dim_num,size_t* global_work_size,size_t* local_work_size,_types...args) {
    oclPtr->kernel = clCreateKernel(program, name, &err);
    if (!oclPtr->kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }
    _create_memory<0,oclPtr>(args...);

    err = clEnqueueNDRangeKernel(oclPtr->commands, oclPtr->kernel, dim_num, NULL,&global_work_size[0] ,&local_work_size[0], 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    clEnqueueTask(oclPtr->commands,oclPtr->kernel,0,NULL,NULL);
}