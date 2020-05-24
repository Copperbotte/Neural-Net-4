
#include "OGLWindow.h"
#include "NNet.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include "cudaNNetProcessor.cuh"

int clamp(int x, int low, int high)
{
    if (x < low) x = low;
    if (high < x) x = high;
    return x;
}

//__constant__ int cuShapeLen[1];
//__constant__ int* cuSigmoidIndices; //hardcoded: 0 is const, 1 is tanh

cudaError cudaNNetProcessor::makeBuffers()
{
    // error check dependent pointers
    if (!_pNet)
    {
        std::cout << "Neural Net not initialized!\n";
        return cudaErrorInitializationError;
    }

    //initialize cuda
    cudaError cudaStatus = cudaSuccess;

    //shape length
    _numWeights = _pNet->getShapeLen() - 1;
    //cudaStatus = cudaMemcpyToSymbol(_cuNumWeights, &_numWeights, sizeof(int), 0, cudaMemcpyHostToDevice);
    //can __constant__ be put in a class?
    cudaStatus = cudaMalloc((void**)&_cuNumWeights, sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        makeMallocError("weight length", cudaStatus);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(_cuNumWeights, &_numWeights, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        makeMemcpyError("weight length", cudaStatus);
        return cudaStatus;
    }

    /*
    //copy sigmoid lookup (hardcoded)
    //The neural net on cuda is already pretty slow.
    //This will add additional overhead that isn't nessisary right now.
    int* cuSigmoidIndices;
    cudaStatus = cudaMalloc((void**)&cuSigmoidIndices, netshapelen * sizeof(int));
    int *SigmoidIndices = new int[netshapelen];
    for (int i = 1; i < netshapelen; ++i)
        SigmoidIndices[i] = 1;
    SigmoidIndices[0] = 0;
    cudaStatus = cudaMemcpy(cuSigmoidIndices, SigmoidIndices, netshapelen * sizeof(int), cudaMemcpyHostToDevice);
    */

    //initialize weights
    _cuWeightData = new float* [_numWeights];
    cudaStatus = cudaMalloc((void**)&_cuWeights, _numWeights * sizeof(cuMatrix));
    if (cudaStatus != cudaSuccess)
    {
        makeMallocError("neural network weights", cudaStatus);
        return cudaStatus;
    }

    for (int i = 0; i < _numWeights; ++i)
    {
        cuMatrix temp = cuMatrix(_pNet->getWeights(i));
        cudaStatus = cudaMemcpy(&_cuWeights[i], &temp, sizeof(cuMatrix), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::string name = "weight matrix " + std::to_string(i);
            makeMemcpyError(name.c_str(), cudaStatus);
            return cudaStatus;
        }

        float* tempData;
        size_t tempDataSize = temp.cols * temp.rows * sizeof(float);
        cudaStatus = cudaMalloc((void**)&tempData, tempDataSize);
        if (cudaStatus != cudaSuccess)
        {
            std::string name = "weight matrix " + std::to_string(i) + " data";
            makeMallocError(name.c_str(), cudaStatus);
            return cudaStatus;
        }

        cudaStatus = cudaMemcpy(tempData, temp.data, tempDataSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::string name = "weight matrix " + std::to_string(i) + " data";
            makeMemcpyError(name.c_str(), cudaStatus);
            return cudaStatus;
        }

        cudaStatus = cudaMemcpy(&(_cuWeights[i].data), &tempData, sizeof(float*), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::string name = "weight matrix " + std::to_string(i) + " data pointer";
            makeMemcpyError(name.c_str(), cudaStatus);
            return cudaStatus;
        }

        _cuWeightData[i] = tempData;
    }

    return cudaStatus;
}

void cudaNNetProcessor::makeMallocError(const char* err, cudaError cudaStatus) const
{
    std::cout << "Failed to allocate " << err << " on gpu!\n";
    std::cout << "error code: " << cudaGetErrorString(cudaStatus);
}

void cudaNNetProcessor::makeMemcpyError(const char* err, cudaError cudaStatus) const
{
    std::cout << "Failed to copy " << err << " to gpu!\n";
    std::cout << "error code: " << cudaGetErrorString(cudaStatus);
}

void cudaNNetProcessor::safeFree(void** ptr)
{
    if (!*ptr) return;
    cudaFree(*ptr);
    *ptr = nullptr;
}

cudaNNetProcessor::cudaNNetProcessor() :
    _cuWeights(nullptr), _cuWeightData(nullptr),
    _numWeights(0), _pNet(nullptr)
{
}

cudaNNetProcessor::~cudaNNetProcessor()
{
    for (int i = 0; i < _numWeights; ++i)
        safeFree((void**)&_cuWeightData[i]);
    safeFree((void**)&_cuWeightData);

    // These pointers are for reference only, and may not lead anywhere
    _pNet = nullptr;

    // This was copied from an nvidia cuda sample.
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
}

cudaNNetProcessor::cudaNNetProcessor(NNet& pNet) :
    _cuWeights(nullptr), _cuWeightData(nullptr),
    _numWeights(0), _pNet(&pNet)
{
    makeBuffers();
}

cudaNNetProcessor::cudaNNetProcessor(const cudaNNetProcessor& N) :
    _cuWeights(nullptr), _cuWeightData(nullptr),
    _numWeights(0), _pNet(N._pNet)
{
    makeBuffers();
}

cudaError cudaNNetProcessor::cudaCopyNNet() const
{
    cudaError cudaStatus = cudaSuccess;
    for (int i = 0; i < _numWeights; ++i)
    {
        cudaStatus = cudaMemcpyAsync(_cuWeightData[i], _pNet->getWeights(i).getDataPtr(),
            _pNet->getWeights(i).getCols() * _pNet->getWeights(i).getRows() * sizeof(float),
            cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::string name = "weight matrix " + std::to_string(i) + " data";
            makeMemcpyError(name.c_str(), cudaStatus);
            return cudaStatus;
        }
    }
    return cudaStatus;
}
