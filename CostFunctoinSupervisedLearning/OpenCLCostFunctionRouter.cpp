//
// Created by Atrin Hojjat on 7/5/16.
//

#include "OpenCLCostFunctionRouter.h"

OpenCLCostFunctionRouter::OpenCLCostFunctionRouter(vector<vector<double>> input, vector<vector<double>> output,
                                                   double learning_rate, double regularization_parameter)
        : input(input),output(output),learning_rate(learning_rate),regularization_parameter(regularization_parameter)
{
}

OpenCLCostFunctionRouter::OpenCLCostFunctionRouter(vector<vector<double>> input, vector<vector<double>> output,
                                                   double learning_rate, double regularization_parameter,
                                                   long max_iters)
        : input(input),output(output),learning_rate(learning_rate),
          regularization_parameter(regularization_parameter),max_iters(max_iters)
{
}

#ifndef EXIT_FAILURE
#define EXIT_FAILURE -1
#endif
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif

void OpenCLCostFunctionRouter::call_func()
{
    long training_input_size  = input.size()*input[0].size();
    long training_output_size = output.size()*output[0].size();
    long training_result_size = weight.size()*weight[0].size();

    int err;                            // error code returned from api calls

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem training_input;
    cl_mem training_output;
    cl_mem training_result;
    cl_mem sizes;

    long sizes_arr[] = {input.size(),input[0].size(),output.size(),output[0].size(),weight.size(),weight[0].size()};


    // Connect to a compute device
    //
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        exit(EXIT_FAILURE);
    }

    // Create a compute context
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        exit(EXIT_FAILURE);
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        exit(EXIT_FAILURE);
    }

    // READ OPENCL FILE

    char** KernelSource;
    ifstream infile("test/data_test02.txt");
    if(!infile.is_open())
        return;
    vector<char*> file(0);
    string line;
    while(std::getline(infile, line)){
        file.insert(file.end(),line);
    }

    program = clCreateProgramWithSource(context, 1, (const char **) &file[0], NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        exit(EXIT_FAILURE);
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    training_input  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(double) * training_input_size , NULL, NULL);
    training_output = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(double) * training_output_size, NULL, NULL);
    training_result = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(double) * training_result_size, NULL, NULL);
    sizes = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(double) * 6, NULL, NULL);
    if (!training_result || !training_output || !training_input || !sizes)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    // Write our data set into the input array in device memory
    //
    err = clEnqueueWriteBuffer(commands, training_input, CL_TRUE, 0, sizeof(double) * training_input_size, &input[0], 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, training_output, CL_TRUE, 0, sizeof(double) * training_output_size, &output[0], 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, sizes, CL_TRUE, 0, sizeof(double) * 6, sizes_arr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &training_input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &training_output);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &training_result);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &sizes);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = training_input_size + training_output_size + training_result_size + 6;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        exit(EXIT_FAILURE);
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer( commands, training_result, CL_TRUE, 0, sizeof(float) * training_result_size, &weight[0], 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    for_each(weight.begin(),weight.end(),[](double x){cout << setw(10) << x << " ";});

    // Shutdown and cleanup
    //
    clReleaseMemObject(training_result);
    clReleaseMemObject(training_output);
    clReleaseMemObject(training_input);
    clReleaseMemObject(sizes);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
}