
#include "OGLWindow.h"
#include "NNet.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "rand.h"

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
    int shape[] = { 2, 3, 1 };
    int shapelen = sizeof(shape) / sizeof(int);

    NNet net = NNet(shapelen, shape);
    net.randomizeNodes(GetTickCount());

    random Random(GetTickCount() + 10000);

    float **i_function = new float*[100];
    float* o_function = new float[100];
    matrix* i_matrix = new matrix[100];
    matrix* o_matrix = new matrix[100];
    for (int i = 0; i < 100; ++i)
    {
        i_function[i] = new float[2];
        float x = Random.xorshfdbl() * 6.0 - 3.0;
        float y = Random.xorshfdbl() * 6.0 - 3.0;
        //float o = 1.0;
        //if ((x+2.0) * (x+2.0) + (y+2.0) * (y+2.0) > 1)
        //    o = -1.0;
        float o = sin(4*x) * sin(4*y);
        i_function[i][0] = x;
        i_function[i][1] = y;
        o_function[i] = o;
        i_matrix[i] = matrix(1, 2, i_function[i]);
        o_matrix[i] = matrix(1, 1, nullptr);
        o_matrix[i].setData(0, 0, o_function[i]);
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

    wnd.setPrintFunc([](const char* str) {std::cout << str << '\n'; });
    wnd.fillColorBuffer(0xFF, 0x8F, 0x00, 0xFF);

    GLFWwindow* window = wnd.init();
    while (wnd.thinkStep())
    {
        for (int i = 0; i < 100; ++i)
        {
            float err = net.backPropArray(i_matrix, o_matrix, 100);
            if(i == 100 - 1)
                std::cout << err << '\n';
        }

        int H = wnd.getHeight();
        int W = wnd.getWidth();
        unsigned char *P = wnd.getColorBufferPtr();

        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                unsigned char *c = (y*W + x)*4 + P;
                float X = ((float)x / (float)W) * 6.0 - 3.0;
                float Y = ((float)y / (float)H) * 6.0 - 3.0;
                float lpi_sample[] = { X,Y };
                matrix lpinput = matrix(1, 2, lpi_sample);
                //float out = sin(X) * sin(Y);
                float out = net.forwardProp(lpinput).getData(0,0);
                out = tanh(out);
                out = (out + 1.0) / 2.0;
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
    }

    for (int i = 0; i < 100; ++i)
        delete[] i_function[i];
    delete[] o_function;
    delete[] i_matrix;
    delete[] o_matrix;

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
