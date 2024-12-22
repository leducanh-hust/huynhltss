#include <iostream>
#include <fstream>
#include <vector>
#include <CL/cl.h>

#define KERNEL_FILE_PATH "kernels/radix.cl"

// Function to load kernel source from file
std::string loadKernelSource(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open kernel file: " << filename << std::endl;
        exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// Function to find the maximum value in the array
int getMax(const std::vector<int>& arr) {
    int max_val = arr[0];
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] > max_val)
            max_val = arr[i];
    }
    return max_val;
}

// Radix sort using OpenCL
void parallelRadixSort(std::vector<int>& arr) {
    size_t n = arr.size();
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    // Initialize OpenCL
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get Platform ID" << std::endl;
        exit(1);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get Device ID" << std::endl;
        exit(1);
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create Context" << std::endl;
        exit(1);
    }

    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create Command Queue" << std::endl;
        exit(1);
    }

    // Load kernel source
    std::string kernel_source = loadKernelSource(KERNEL_FILE_PATH);
    const char* source_cstr = kernel_source.c_str();
    size_t source_size = kernel_source.size();

    program = clCreateProgramWithSource(context, 1, &source_cstr, &source_size, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create Program from source" << std::endl;
        exit(1);
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::cerr << "Build program error:\n" << buffer << std::endl;
        exit(1);
    }

    kernel = clCreateKernel(program, "radix_sort", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create Kernel" << std::endl;
        exit(1);
    }

    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create input buffer" << std::endl;
        exit(1);
    }

    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create output buffer" << std::endl;
        exit(1);
    }

    cl_mem histogram_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create histogram buffer" << std::endl;
        exit(1);
    }

    err = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, n * sizeof(int), arr.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to write to input buffer" << std::endl;
        exit(1);
    }

    int max_val = getMax(arr);
    for (int bit = 0; (max_val >> bit) > 0; ++bit) {
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &histogram_buffer);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &bit);
        err |= clSetKernelArg(kernel, 4, sizeof(int), &n);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to set Kernel arguments, error code: " << err << std::endl;
            exit(1);
        }

        size_t global_size = n;
        size_t local_size = 64;
        if (global_size % local_size != 0) {
            global_size = ((global_size / local_size) + 1) * local_size;
        }

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue Kernel, error code: " << err << std::endl;
            exit(1);
        }

        clFinish(queue);

        cl_mem temp = input_buffer;
        input_buffer = output_buffer;
        output_buffer = temp;
    }

    err = clEnqueueReadBuffer(queue, input_buffer, CL_TRUE, 0, n * sizeof(int), arr.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to read from input buffer" << std::endl;
        exit(1);
    }

    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(histogram_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main() {
    const size_t n = 1000000;
    std::vector<int> arr(n);

    srand(static_cast<unsigned>(time(0)));
    for (size_t i = 0; i < n; ++i) {
        arr[i] = rand() % 1000000;
    }

    clock_t start_time = clock();
    parallelRadixSort(arr);
    clock_t end_time = clock();

    std::cout << "Execution time: " << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC << " seconds\n";

    return 0;
}
