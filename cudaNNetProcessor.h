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
public:
    unsigned char* _cuColorBuffer;

    cuMatrix* _cuWeights;
    float** _cuWeightData;
    unsigned int _numWeights;
    int* _cuNumWeights;

    NNet* _pNet;
    const OGLWindow* _pWnd;

    dim3 _grid;

    void makeBuffers();

//public:

    cudaNNetProcessor();
    ~cudaNNetProcessor();
    cudaNNetProcessor(NNet& pNet, const OGLWindow& wnd);
    cudaNNetProcessor(const cudaNNetProcessor& N);
    
    cudaError cudaCopyNNet() const;

    //problem specific functions
    //should this be in a child class?
    cudaError cudaRenderNNet() const;

};
