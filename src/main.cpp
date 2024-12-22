#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#define FILTER_SIZE 3 // Median filter size

// Utility function to load the kernel file
std::string loadKernel(const std::string &filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open kernel file: " + filePath);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int main()
{
    try
    {
        // Load input image
        cv::Mat inputImage = cv::imread("input/input.jpg", cv::IMREAD_COLOR);
        if (inputImage.empty())
        {
            throw std::runtime_error("Failed to load input image!");
        }
        int width = inputImage.cols;
        int height = inputImage.rows;
        int channels = inputImage.channels();
        cv::Mat outputImage(height, width, CV_8UC3);

        // OpenCL Setup
        cl_platform_id platform;
        cl_device_id device;
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_kernel kernel;

        // Get platform and device
        clGetPlatformIDs(1, &platform, nullptr);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueueWithProperties(context, device, nullptr, nullptr);

        // Load and compile kernel
        std::string kernelSource = loadKernel("kernels/median_filter.cl");
        const char *source = kernelSource.c_str();
        program = clCreateProgramWithSource(context, 1, &source, nullptr, nullptr);
        clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        kernel = clCreateKernel(program, "medianFilter", nullptr);

        // Buffers
        size_t imageSize = width * height * channels;
        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, imageSize, nullptr, nullptr);
        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imageSize, nullptr, nullptr);

        // Write input image to GPU
        clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, imageSize, inputImage.data, 0, nullptr, nullptr);

        // Set kernel arguments
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
        clSetKernelArg(kernel, 2, sizeof(int), &width);
        clSetKernelArg(kernel, 3, sizeof(int), &height);
        clSetKernelArg(kernel, 4, sizeof(int), &channels);
        int filterSize = FILTER_SIZE; // Create a non-const local variable
        clSetKernelArg(kernel, 5, sizeof(int), &filterSize);

        // Run kernel
        size_t globalSize[2] = {(size_t)width, (size_t)height};
        clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
        clFinish(queue);

        // Read back output
        clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, imageSize, outputImage.data, 0, nullptr, nullptr);

        // Save and show result
        cv::imwrite("output/output.jpg", outputImage);
        cv::imshow("Filtered Image", outputImage);
        cv::waitKey(0);

        // Cleanup
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
