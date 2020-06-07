#pragma once

#include "OGLWindow.h"
#include "matrix.h"
#include "NNet.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

namespace
{
    struct cuMatrix
    {
        int cols, rows;
        float* data;

        __device__ cuMatrix();
        __device__ __host__ ~cuMatrix();
        __device__ cuMatrix(int Cols, int Rows);
        __device__ cuMatrix(int Cols, int Rows, const float* Data);
        __device__ __host__ cuMatrix(const cuMatrix& R);
        __host__ cuMatrix(const matrix& R);

        __device__ __host__ cuMatrix& operator=(const cuMatrix& R);
        __device__ cuMatrix operator*(const cuMatrix& R) const;

        __device__ float getData(int c, int r) const;
        __device__ void setData(int c, int r, float d);

        __device__ int getDataCount() const;
    };
}

class cudaNNetProcessor
{
protected:
    //cuMatrix* _cuNodes; // can be floats
    //float** _cuNodeData;

    cuMatrix* _cuWeights;
    float** _cuWeightData;
    unsigned int _numWeights;
    int* _cuNumWeights;

    NNet* _pNet;

    cudaError makeBuffers();
    //makeMatrix won't link, because the above cuMatrix is defined in two spots.
    //This is because cuda requires all the device code to be in the same file.
    //Either c++ wont link, or cuda won't.
    cudaError makeMatrix(cuMatrix* output, float** outputData, const cuMatrix& input, const std::string name);
    void makeMallocError(const char* err, cudaError cudaStatus) const;
    void makeMemcpyError(const char* err, cudaError cudaStatus) const;
    void makeKernelError(const char* err, cudaError cudaStatus) const;
    void safeFreeArray(void** ptr);
    void cuSafeFree(void** ptr);

public:

    cudaNNetProcessor();
    ~cudaNNetProcessor();
    cudaNNetProcessor(NNet& pNet);
    cudaNNetProcessor(const cudaNNetProcessor& N);

    //cudaError cudaCopyInputNode() const;
    cudaError cudaCopyNNet() const;
    
    //cudaError cudaForwardPropStep(int n);
    cudaError cudaForwardPropArray(void* cuFinalNode, void* cuInitialNode,
        int cuArrayCount, dim3 &threadGrid, dim3 &blockGrid) const;
};
