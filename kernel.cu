
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

struct cuMatrix
{
    int cols, rows;
    float* data;

    __device__ cuMatrix() :cols(1), rows(1)
    {
        data = new float[1];
        data[0] = 1.0;
    }

    __device__ __host__ ~cuMatrix()
    {
        delete[] data;
        cols = rows = 0;
        data = nullptr;
    }

    __device__ cuMatrix(int Cols, int Rows):cols(Cols),rows(Rows)
    {
        data = new float[cols * rows];
        for (int i = 0; i < cols * rows; ++i)
            data[i] = 0.0f;
    }

    __device__ cuMatrix(int Cols, int Rows, const float* Data) : cols(Cols), rows(Rows)
    {
        data = new float[cols * rows];
        memcpy(data, Data, cols * rows * sizeof(float));
    }

    __device__ __host__ cuMatrix(const cuMatrix& R) : cols(R.cols), rows(R.rows)
    {
        data = new float[cols * rows];
        memcpy(data, R.data, cols * rows * sizeof(float));
    }

    __host__ cuMatrix(const matrix& R) :cols(R.getCols()), rows(R.getRows())
    {
        data = new float[cols * rows];
        memcpy(data, R.getDataPtr(), cols * rows * sizeof(float));
    }

    __device__ __host__ cuMatrix& operator=(const cuMatrix& R)
    {
        delete[] data;
        cols = R.cols;
        rows = R.rows;
        data = new float[cols * rows];
        memcpy(data, R.data, rows * cols * sizeof(float));
        return *this;
    }

    __device__ cuMatrix operator*(const cuMatrix& R) const
    {
        cuMatrix M(R.cols, rows);
        for (int c = 0; c < M.cols; ++c)
            for (int r = 0; r < M.rows; ++r)
                for (int n = 0; n < cols; ++n)
                    M.setData(c, r, M.getData(c, r) + getData(n, r) * R.getData(c, n));
        return M;
    }

    __device__ float getData(int c, int r) const
    {
        return data[r * cols + c];
    }

    __device__ void setData(int c, int r, float d)
    {
        data[r * cols + c] = d;
    }

};

__constant__ int cuShapeLen[1];
//__constant__ int* cuSigmoidIndices; //hardcoded: 0 is const, 1 is tanh

__device__ cuMatrix cuForwardProp(cuMatrix &input, const cuMatrix *cuWeights)
{
    //this should have as much static memory as possible
    cuMatrix node = input;
    
    for (int i = 0; i < cuShapeLen[0] - 1; ++i)
    {
        cuMatrix bias = cuMatrix(1, node.rows + 1);
        memcpy(&bias.data[1], node.data, node.rows * sizeof(float));
        bias.setData(0, 0, 1.0f);

        for (int r = 1; r < node.rows; ++r)
        {
            //apply sigmoid
            float sig = bias.getData(0, r);
            //if (cuSigmoidIndices[i] == 1)
            if(0 < i)
                sig = tanhf(sig);
            bias.setData(0, r, sig);
        }

        //forward propogate
        node = cuWeights[i] * bias;
    }

    return node;
}

__global__ void kernel(unsigned char* colorBuffer, cuMatrix *cuWeights)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    float X = (((float)x / (float)gridDim.x) * 2.0 - 1.0) * 3.0;
    float Y = (((float)y / (float)gridDim.y) * 2.0 - 1.0) * 3.0;

    float initialNode[2] = { X, Y };
    float result = cuForwardProp(cuMatrix(1, 2, initialNode), cuWeights).getData(0, 0);
    
    result = tanhf(result);
    result = (result + 1.0f) / 2.0f;

    //int icolor = (unsigned char)(result * 255.0f);
    //if (255 < icolor) icolor = 255;
    unsigned char color = (unsigned char)(result * 255.0f);
    unsigned char* px = colorBuffer + 4 * offset;
    px[0] = color;
    px[1] = color;
    px[2] = color;
    px[3] = 0xFF;
}

__global__ void debugMatrixKernel(unsigned char* colorBuffer, cuMatrix* cuWeights)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    float X = (((float)x / (float)gridDim.x) * 2.0 - 1.0) * 5.0 + 5.0;
    float Y = (((float)y / (float)gridDim.y) * 2.0 - 1.0) * 5.0 * (3.0 / 4.0) + 3.0;

    float pi = 3.141592;
    float result = sinf(X * pi) * sinf(X * pi) * sinf(Y * pi) * sinf(Y * pi);
    result = powf(result, 0.1);

    float initialNode[3] = { 1.0, X, Y };
    cuMatrix matrixTest = cuWeights[0] * cuMatrix(1, 3, initialNode);
    result = matrixTest.getData(0, 0);

    //matrixTest = cuMatrix(1, 2, initialNode);

    /*/
    //Array content debugging
    if (0.0 < X && X < matrixTest.cols)
        if (0.0 < Y && Y < matrixTest.rows)
            result = matrixTest.getData((int)X, (int)Y);

    //*/
    /*
    //Array size debugging
    if (0.0 < X && X < matrixTest.cols)
        if (0.0 < Y && Y < matrixTest.rows)
            result = 0.0f;
    //*/

    //Array count debugging
    /*
    if (0.0 < X && X < cuShapeLen[0])
        if (0.0 < Y && Y < 1.0)
            result = 0.0f;
    */

    result = tanhf(result);
    result = (result + 1.0f) / 2.0f;

    int icolor = (unsigned char)(result * 255.0f);
    if (255 < icolor) icolor = 255;
    unsigned char color = icolor;
    unsigned char* px = colorBuffer + 4 * offset;
    px[0] = color;
    px[1] = color;
    px[2] = color;
    px[3] = 0xFF;
}

int main()
{
    int shape[] = { 2, 3, 1 };
    //int shape[] = { 2, 1 };
    int shapelen = sizeof(shape) / sizeof(int);

    //NNet net = NNet();
    NNet net = NNet(shapelen, shape);
    net.randomizeNodes(GetTickCount());

    //generate test data
    random Random(GetTickCount() + 10000);

    float **i_function = new float*[100];
    float* o_function = new float[100];
    matrix* i_matrix = new matrix[100];
    matrix* o_matrix = new matrix[100];
    float (*test_func)(float, float) = [](float x, float y)
    {
        float X = x - 1;
        if (X * X + y * y < 4.0) return 1.0f;
        return -1.0f;
        //return sin(2.0f * x) * sin(2.0f * y);
        //return x + y - 2;
    };
    for (int i = 0; i < 100; ++i)
    {
        i_function[i] = new float[2];
        float x = Random.xorshfdbl() * 6.0 - 3.0;
        float y = Random.xorshfdbl() * 6.0 - 3.0;
        float o = test_func(x, y);
        i_function[i][0] = x;
        i_function[i][1] = y;
        o_function[i] = o;
        i_matrix[i] = matrix(1, 2, i_function[i]);
        o_matrix[i] = matrix(1, 1, nullptr);
        o_matrix[i].setData(0, 0, o_function[i]);
    }
    /*
    for (int i = 0; i < 1000; ++i)
    {
        net.backPropArray(i_matrix, o_matrix, 100);
        if (i % 100 == 0)
            std::cout << i << '\n';
    } */
    
    //create window
    OGLWindow wnd("NN4");

    wnd.setPrintFunc([](const char* str) {std::cout << str << '\n'; });
    wnd.fillColorBuffer(0xFF, 0x8F, 0x00, 0xFF);

    GLFWwindow* window = wnd.init();

    //initialize cuda
    unsigned char* dev_bmp;
    cudaError cudaStatus = cudaSuccess;
    cudaStatus = cudaMalloc((void**)&dev_bmp, wnd.getWidth() * wnd.getHeight() * 4 * sizeof(char));
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Failed to allocate bitmap on gpu\n";
        std::cout << "error code: " << cudaGetErrorString(cudaStatus);
        for (int i = 0; i < 100; ++i)
            delete[] i_function[i];
        delete[] o_function;
        delete[] i_matrix;
        delete[] o_matrix;
        return;
    }

    
    //copy shape length
    int netshapelen = net.getShapeLen();
    //cudaStatus = cudaMalloc((void**)&cuShapeLen, sizeof(int));
    cudaStatus = cudaMemcpyToSymbol(cuShapeLen, &netshapelen, sizeof(int), 0, cudaMemcpyHostToDevice);

    /*
    //copy sigmoid lookup (hardcoded)
    int* cuSigmoidIndices;
    cudaStatus = cudaMalloc((void**)&cuSigmoidIndices, netshapelen * sizeof(int));
    int *SigmoidIndices = new int[netshapelen];
    for (int i = 1; i < netshapelen; ++i)
        SigmoidIndices[i] = 1;
    SigmoidIndices[0] = 0;
    cudaStatus = cudaMemcpy(cuSigmoidIndices, SigmoidIndices, netshapelen * sizeof(int), cudaMemcpyHostToDevice);
    */

    //initialize weights
    cuMatrix* cuWeights;
    float** cuDataPtr = new float* [netshapelen - 1];
    cudaStatus = cudaMalloc((void**)&cuWeights, (netshapelen - 1) * sizeof(cuMatrix));
    for (int i = 0; i < netshapelen - 1; ++i)
    {
        cuMatrix temp = cuMatrix(net.getWeights(i));
        cudaStatus = cudaMemcpy(&cuWeights[i], &temp, sizeof(cuMatrix), cudaMemcpyHostToDevice);

        float* tempData;
        size_t tempDataSize = temp.cols * temp.rows * sizeof(float);
        cudaStatus = cudaMalloc((void**)&tempData, tempDataSize);
        cudaStatus = cudaMemcpy(tempData, temp.data, tempDataSize, cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpy(&(cuWeights[i].data), &tempData, sizeof(float*), cudaMemcpyHostToDevice);
        cuDataPtr[i] = tempData;
    }

    dim3 grid(wnd.getWidth(), wnd.getHeight());
    
    net.setRate(0.5);

    while (wnd.thinkStep())
    {
        float err = 0.0f;
        for (int i = 0; i < 100; ++i)
            err = net.backPropArray(i_matrix, o_matrix, 100);
        std::cout << err << '\n';
        
        //copy weights
        for (int i = 0; i < net.getShapeLen() - 1; ++i)
        {
            cudaStatus = cudaMemcpy(cuDataPtr[i], net.getWeights(i).getDataPtr(),
                net.getWeights(i).getCols() * net.getWeights(i).getRows() * sizeof(float),
                cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
            {
                std::cout << "Failed to copy matrix memory to gpu\n";
                std::cout << "error code : " << cudaGetErrorString(cudaStatus);
                break;
            }
        }
        
        //render screen
        kernel <<<grid, 1 >>> (dev_bmp, cuWeights);
        cudaStatus = cudaMemcpy(wnd.getColorBufferPtr(), dev_bmp,
            wnd.getWidth() * wnd.getHeight() * 4 * sizeof(char), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            std::cout << "Failed to copy memory from gpu\n";
            std::cout << "error code : " << cudaGetErrorString(cudaStatus);
            break;
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

    cudaFree(dev_bmp);

    //cudaFree(cuSigmoidIndices);
    for (int i = 0; i < net.getShapeLen() - 1; ++i)
        cudaFree(cuDataPtr[i]);
    delete[] cuDataPtr;
    cudaFree(cuWeights);
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

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
