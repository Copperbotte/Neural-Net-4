#pragma once

#include "OGLWindow.h"
#include "matrix.h"
#include "NNet.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

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
};

class cudaNNetProcessor
{
protected:
    cuMatrix* _cuWeights;
    float** _cuWeightData;
    unsigned int _numWeights;
    int* _cuNumWeights;

    NNet* _pNet;

    cudaError makeBuffers();
    void makeMallocError(const char* err, cudaError cudaStatus) const;
    void makeMemcpyError(const char* err, cudaError cudaStatus) const;
    void safeFree(void** ptr);

public:

    cudaNNetProcessor();
    ~cudaNNetProcessor();
    cudaNNetProcessor(NNet& pNet);
    cudaNNetProcessor(const cudaNNetProcessor& N);

    cudaError cudaCopyNNet() const; 
};
