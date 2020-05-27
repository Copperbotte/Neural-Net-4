
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

    /*
    //initialize nodes
    _cuNodeData = new float* [_numWeights + 1];
    cudaStatus = cudaMalloc((void**)&_cuNodes, (_numWeights + 1) * sizeof(cuMatrix));
    if (cudaStatus != cudaSuccess)
    {
        makeMallocError("neural network nodes", cudaStatus);
        return cudaStatus;
    }

    for (int i = 0; i < _numWeights + 1; ++i)
    {
        cuMatrix temp = cuMatrix(1, _pNet->getShape(i));
        cudaStatus = cudaMemcpy(&_cuNodes[i], &temp, sizeof(cuMatrix), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::string name = "node matrix " + std::to_string(i);
            makeMemcpyError(name.c_str(), cudaStatus);
            return cudaStatus;
        }

        float* tempData;
        size_t tempDataSize = temp.cols * temp.rows * sizeof(float);
        cudaStatus = cudaMalloc((void**)&tempData, tempDataSize);
        if (cudaStatus != cudaSuccess)
        {
            std::string name = "node matrix " + std::to_string(i) + " data";
            makeMallocError(name.c_str(), cudaStatus);
            return cudaStatus;
        }

        //no need to copy node data, except for the initial node for forward props
        //cudaStatus = cudaMemcpy(tempData, temp.data, tempDataSize, cudaMemcpyHostToDevice);
        //if (cudaStatus != cudaSuccess)
        //{
        //    std::string name = "node matrix " + std::to_string(i) + " data";
        //    makeMemcpyError(name.c_str(), cudaStatus);
        //    return cudaStatus;
        //}

        cudaStatus = cudaMemcpy(&(_cuNodes[i].data), &tempData, sizeof(float*), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::string name = "node matrix " + std::to_string(i) + " data pointer";
            makeMemcpyError(name.c_str(), cudaStatus);
            return cudaStatus;
        }

        _cuNodeData[i] = tempData;
    }
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
        makeMatrix(&_cuWeights[i], &_cuWeightData[i],
            cuMatrix(_pNet->getWeights(i)), "weight matrix " + std::to_string(i));
    
    return cudaStatus;
}

cudaError cudaNNetProcessor::makeMatrix(cuMatrix* output, float** outputData, const cuMatrix& input, const std::string name)
{
    cudaError cudaStatus = cudaSuccess;

    cudaStatus = cudaMemcpy(output, &input, sizeof(cuMatrix), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        makeMemcpyError(name.c_str(), cudaStatus);
        return cudaStatus;
    }

    float* tempData;
    size_t tempDataSize = input.cols * input.rows * sizeof(float);
    cudaStatus = cudaMalloc((void**)&tempData, tempDataSize);
    if (cudaStatus != cudaSuccess)
    {
        makeMallocError((name + " data").c_str(), cudaStatus);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(tempData, input.data, tempDataSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        makeMemcpyError((name + " data").c_str(), cudaStatus);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(&(output->data), &tempData, sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        makeMemcpyError((name + " data pointer").c_str(), cudaStatus);
        return cudaStatus;
    }

    *outputData = tempData;

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

void cudaNNetProcessor::safeFreeArray(void** ptr)
{
    if (!*ptr) return;
    delete[] *ptr;
    *ptr = nullptr;
}

void cudaNNetProcessor::cuSafeFree(void** ptr)
{
    if (!*ptr) return;
    cudaFree(*ptr);
    *ptr = nullptr;
}

cudaNNetProcessor::cudaNNetProcessor() :
    //_cuNodes(nullptr), _cuNodeData(nullptr),
    _cuWeights(nullptr), _cuWeightData(nullptr),
    _numWeights(0), _pNet(nullptr)
{
}

cudaNNetProcessor::~cudaNNetProcessor()
{
    for (int i = 0; i < _numWeights; ++i)
        cuSafeFree((void**)&_cuWeightData[i]);
    safeFreeArray((void**)&_cuWeightData);

    cuSafeFree((void**)&_cuWeights);
    cuSafeFree((void**)&_cuNumWeights);

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
    //_cuNodes(nullptr), _cuNodeData(nullptr),
    _cuWeights(nullptr), _cuWeightData(nullptr),
    _numWeights(0), _pNet(&pNet)
{
    makeBuffers();
}

cudaNNetProcessor::cudaNNetProcessor(const cudaNNetProcessor& N) :
    //_cuNodes(nullptr), _cuNodeData(nullptr),
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

//Cuda device code
//these decorated functions appear to have to not sit in a class structure.
//Maybe they can be wrapped?

namespace
{
    __device__ cuMatrix::cuMatrix() :cols(1), rows(1)
    {
        data = new float[1];
        data[0] = 1.0;
    }

    __device__ __host__ cuMatrix::~cuMatrix()
    {
        delete[] data;
        cols = rows = 0;
        data = nullptr;
    }

    __device__ cuMatrix::cuMatrix(int Cols, int Rows) :cols(Cols), rows(Rows)
    {
        data = new float[cols * rows];
        for (int i = 0; i < cols * rows; ++i)
            data[i] = 0.0f;
    }

    __device__ cuMatrix::cuMatrix(int Cols, int Rows, const float* Data) : cols(Cols), rows(Rows)
    {
        data = new float[cols * rows];
        memcpy(data, Data, cols * rows * sizeof(float));
    }

    __device__ __host__ cuMatrix::cuMatrix(const cuMatrix& R) : cols(R.cols), rows(R.rows)
    {
        data = new float[cols * rows];
        memcpy(data, R.data, cols * rows * sizeof(float));
    }

    __host__ cuMatrix::cuMatrix(const matrix& R) :cols(R.getCols()), rows(R.getRows())
    {
        data = new float[cols * rows];
        memcpy(data, R.getDataPtr(), cols * rows * sizeof(float));
    }

    __device__ __host__ cuMatrix& cuMatrix::operator=(const cuMatrix& R)
    {
        delete[] data;
        cols = R.cols;
        rows = R.rows;
        data = new float[cols * rows];
        memcpy(data, R.data, rows * cols * sizeof(float));
        return *this;
    }

    __device__ cuMatrix cuMatrix::operator*(const cuMatrix& R) const
    {
        cuMatrix M(R.cols, rows);
        for (int c = 0; c < M.cols; ++c)
            for (int r = 0; r < M.rows; ++r)
                for (int n = 0; n < cols; ++n)
                    M.setData(c, r, M.getData(c, r) + getData(n, r) * R.getData(c, n));
        return M;
    }

    __device__ float cuMatrix::getData(int c, int r) const
    {
        return data[r * cols + c];
    }

    __device__ void cuMatrix::setData(int c, int r, float d)
    {
        data[r * cols + c] = d;
    }

    __device__ cuMatrix cuForwardProp(cuMatrix& input, const cuMatrix* cuWeights, int* cuNumWeights)
    {
        //this should have as much static memory as possible
        cuMatrix node = input;

        for (int i = 0; i < *cuNumWeights; ++i)
        {
            cuMatrix bias = cuMatrix(1, node.rows + 1);
            memcpy(&bias.data[1], node.data, node.rows * sizeof(float));
            bias.setData(0, 0, 1.0f);

            for (int r = 1; r < node.rows; ++r)
            {
                //apply sigmoid
                float sig = bias.getData(0, r);
                //if (cuSigmoidIndices[i] == 1)
                if (0 < i)
                    sig = tanhf(sig);
                bias.setData(0, r, sig);
            }

            //forward propogate
            node = cuWeights[i] * bias;
        }

        return node;
    }
}
