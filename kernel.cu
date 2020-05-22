
#include "OGLWindow.h"
#include "NNet.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

int clamp(int x, int low, int high)
{
    if (x < low) x = low;
    if (high < x) x = high;
    return x;
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    NNet net = NNet();
    net.randomizeNodes();

    float i_sample[] = { 1.0, 1.0 };
    float o_sample[] = { 0.5 };
    matrix input = matrix(1, 2, i_sample);
    matrix expected = matrix(1, 1, o_sample);
    matrix output = net.forwardProp(input);

    for (int i = 0; i < 100000; ++i)
    {
        float err = net.backProp(input, expected);
        if(i % 1000 == 0)
            std::cout << i << " : " << err << '\n';
    }

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    OGLWindow wnd("NN4");

    wnd.extra = &net;

    wnd.setPrintFunc([](const char* str) {std::cout << str << '\n'; });
    wnd.fillColorBuffer(0xFF, 0x8F, 0x00, 0xFF);

    GLFWwindow* window = wnd.init();
    wnd.setThinkFunc([](OGLWindow* This, double time)
        {
            GLFWwindow* window = This->getWindowPtr();
            
            NNet *Net = (NNet*)This->extra;
            /*
            float li_sample[] = { 1.0, 1.0 };
            float lo_sample[] = { 0.5 };
            matrix linput = matrix(1, 2, li_sample);
            matrix lexpected = matrix(1, 1, lo_sample);
            float err = Net->backProp(linput, lexpected);
            std::cout << err << '\n';
            */

            int H = This->getHeight();
            int W = This->getWidth();
            unsigned char *P = This->getColorBufferPtr();

            for (int y = 0; y < H; ++y)
            {
                for (int x = 0; x < W; ++x)
                {
                    unsigned char *c = (y*W + x)*4 + P;
                    float X = 2.0 * (float)x / (float)W;
                    float Y = 2.0 * (float)y / (float)H;
                    float lpi_sample[] = { X,Y };
                    matrix lpinput = matrix(1, 2, lpi_sample);
                    float out = Net->forwardProp(lpinput).getData(0,0);
                    if (out < 0.0f) out = 0.0f;
                    out = tanh(out);
                    //if (1.0f < out) out = 1.0f;
                    unsigned char color = (int)(out * 255.0f);
                    c[0] = color;
                    c[1] = color;
                    c[2] = color;
                    c[3] = 0xFF;
                }
            }

            //double mx, my;
            //glfwGetCursorPos(window, &mx, &my);

            //my = height - my;

            //mx = clamp(mx, 0, width - 1);
            //my = clamp(my, 0, height - 1);

            //float fm[2] = { (float)mx / width, (float)my / height };
        });
    wnd.think();

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
