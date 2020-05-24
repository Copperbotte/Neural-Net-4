
#include "OGLWindow.h"
#include "NNet.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include "cudaNNetProcessor.h"

int clamp(int x, int low, int high)
{
    if (x < low) x = low;
    if (high < x) x = high;
    return x;
}

//__constant__ int cuShapeLen[1];
//__constant__ int* cuSigmoidIndices; //hardcoded: 0 is const, 1 is tanh

//these decorated functions appear to have to not sit in a class structure.
//Maybe they can be wrapped?
__device__ cuMatrix cuForwardProp(cuMatrix &input, const cuMatrix *cuWeights, int *cuNumWeights)
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
            if(0 < i)
                sig = tanhf(sig);
            bias.setData(0, r, sig);
        }

        //forward propogate
        node = cuWeights[i] * bias;
    }

    return node;
}

__global__ void kernel(unsigned char* colorBuffer, cuMatrix *cuWeights, int *cuNumWeights)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    float X = (((float)x / (float)gridDim.x) * 2.0 - 1.0) * 3.0;
    float Y = (((float)y / (float)gridDim.y) * 2.0 - 1.0) * 3.0;

    float initialNode[2] = { X, Y };
    float result = cuForwardProp(cuMatrix(1, 2, initialNode), cuWeights, cuNumWeights).getData(0, 0);
    
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

__global__ void debugMatrixKernel(unsigned char* colorBuffer, cuMatrix* cuWeights, int* cuNumWeights)
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

void cudaNNetProcessor::makeBuffers()
{
    void (*makeMallocError)(const char*, cudaError) = [](const char* err, cudaError cudaStatus)
    {
        std::cout << "Failed to allocate " << err << " on gpu!\n";
        std::cout << "error code: " << cudaGetErrorString(cudaStatus);
    };

    void (*makeMemcpyError)(const char*, cudaError) = [](const char* err, cudaError cudaStatus)
    {
        std::cout << "Failed to copy " << err << " to gpu!\n";
        std::cout << "error code: " << cudaGetErrorString(cudaStatus);
    };


    // error check dependent pointers
    if (!_pNet)
    {
        std::cout << "Neural Net not initialized!\n";
        return;
    }

    if (!_pWnd)
    {
        std::cout << "Window not initialized!\n";
        return;
    }

    //initialize cuda
    cudaError cudaStatus = cudaSuccess;

    //color buffer
    cudaStatus = cudaMalloc((void**)&_cuColorBuffer, _pWnd->getWidth() * _pWnd->getHeight() * 4 * sizeof(char));
    if (cudaStatus != cudaSuccess)
    {
        makeMallocError("bitmap", cudaStatus);
        return;
    }

    //shape length
    _numWeights = _pNet->getShapeLen() - 1;
    //cudaStatus = cudaMemcpyToSymbol(_cuNumWeights, &_numWeights, sizeof(int), 0, cudaMemcpyHostToDevice);
    //can __constant__ be put in a class?
    cudaStatus = cudaMalloc((void**)&_cuNumWeights, sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        makeMallocError("weight length", cudaStatus);
        return;
    }

    cudaStatus = cudaMemcpy(_cuNumWeights, &_numWeights, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        makeMemcpyError("weight length", cudaStatus);
        return;
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
        return;
    }
    
    for (int i = 0; i < _numWeights; ++i)
    {
        cuMatrix temp = cuMatrix(_pNet->getWeights(i));
        cudaStatus = cudaMemcpy(&_cuWeights[i], &temp, sizeof(cuMatrix), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::string name = "weight matrix " + std::to_string(i);
            makeMemcpyError(name.c_str(), cudaStatus);
            return;
        }

        float* tempData;
        size_t tempDataSize = temp.cols * temp.rows * sizeof(float);
        cudaStatus = cudaMalloc((void**)&tempData, tempDataSize);
        if (cudaStatus != cudaSuccess)
        {
            std::string name = "weight matrix " + std::to_string(i) + " data";
            makeMallocError(name.c_str(), cudaStatus);
            return;
        }

        cudaStatus = cudaMemcpy(tempData, temp.data, tempDataSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::string name = "weight matrix " + std::to_string(i) + " data";
            makeMemcpyError(name.c_str(), cudaStatus);
            return;
        }

        cudaStatus = cudaMemcpy(&(_cuWeights[i].data), &tempData, sizeof(float*), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::string name = "weight matrix " + std::to_string(i) + " data pointer";
            makeMemcpyError(name.c_str(), cudaStatus);
            return;
        }

        _cuWeightData[i] = tempData;
    }
    
    _grid = dim3(_pWnd->getWidth(), _pWnd->getHeight());
}

cudaNNetProcessor::cudaNNetProcessor() :
    _cuColorBuffer(nullptr), _cuWeights(nullptr),
    _cuWeightData(nullptr), _numWeights(0),
    _pNet(nullptr), _pWnd(nullptr)
{
}

cudaNNetProcessor::~cudaNNetProcessor()
{
    void (*safeFree)(void**) = [](void** ptr) {
        if (!*ptr) return;
        cudaFree(*ptr);
        *ptr = nullptr;
    };

    for (int i = 0; i < _numWeights; ++i)
        safeFree((void**)&_cuWeightData[i]);
    safeFree((void**)&_cuWeightData);
    safeFree((void**)&_cuColorBuffer);

    // These pointers are for reference only, and may not lead anywhere
    _pNet = nullptr;
    _pWnd = nullptr;

    // This was copied from an nvidia cuda sample.
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
}

cudaNNetProcessor::cudaNNetProcessor(NNet& pNet, const OGLWindow& wnd) :
    _cuColorBuffer(nullptr), _cuWeights(nullptr),
    _cuWeightData(nullptr), _numWeights(0),
    _pNet(&pNet), _pWnd(&wnd)
{
    makeBuffers();
}

cudaNNetProcessor::cudaNNetProcessor(const cudaNNetProcessor& N) :
    _cuColorBuffer(nullptr), _cuWeights(nullptr),
    _cuWeightData(nullptr), _numWeights(0),
    _pNet(N._pNet), _pWnd(N._pWnd)
{
    makeBuffers();
}

cudaError cudaNNetProcessor::cudaCopyNNet() const
{
    //this is now used in two spots, I ought to make it a full function later.
    void (*makeMemcpyError)(const char*, cudaError) = [](const char* err, cudaError cudaStatus)
    {
        std::cout << "Failed to copy " << err << " to gpu!\n";
        std::cout << "error code: " << cudaGetErrorString(cudaStatus);
    };

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

cudaError cudaNNetProcessor::cudaRenderNNet() const
{
    cudaError cudaStatus = cudaSuccess;
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Failed to synchronize gpu!\n";
        std::cout << "error code : " << cudaGetErrorString(cudaStatus);
        return cudaStatus;
    }

    //render screen
    kernel <<<_grid, 1 >>> (_cuColorBuffer, _cuWeights, _cuNumWeights);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Failure in gpu kernel!\n";
        std::cout << "error code : " << cudaGetErrorString(cudaStatus);
        return cudaStatus;
    }


    cudaStatus = cudaMemcpy(_pWnd->getColorBufferPtr(), _cuColorBuffer,
        _pWnd->getWidth() * _pWnd->getHeight() * 4 * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Failed to copy memory from gpu\n";
        std::cout << "error code : " << cudaGetErrorString(cudaStatus);
        return cudaStatus;
    }

    return cudaStatus;
}

