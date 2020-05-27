
#include "cudaNNetRenderer.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>

//Cuda requires(?) all device code to be in the same file. This device code currently sits at the bottom of this file.
__global__ void kernel(unsigned char* colorBuffer, cuMatrix* cuWeights, int* cuNumWeights);
__global__ void testKernel(unsigned char* colorBuffer, cuMatrix* cuWeights, int* cuNumWeights);

__global__ void keForwardProp(cuMatrix* cuFinalNode, cuMatrix* cuWeights, int* cuNumWeights);
__global__ void keDisplayNodes(unsigned char* cuColorBuffer, cuMatrix* cuFinalNode);

cudaError cudaNNetRenderer::makeBuffers()
{
	cudaError cudaStatus = cudaSuccess;
    if (!_pNet)
    {
        std::cout << "Neural Net not initialized!\n";
        return cudaErrorInitializationError;
    }

    if (!_pWnd)
    {
        std::cout << "Window not initialized!\n";
        return cudaErrorInitializationError;
    }
    
    //buffers
    int pxCount = _pWnd->getWidth() * _pWnd->getHeight();

	//color buffer
	cudaStatus = cudaMalloc((void**)&_cuColorBuffer, pxCount * 4 * sizeof(char));
	if (cudaStatus != cudaSuccess)
	{
		makeMallocError("bitmap", cudaStatus);
		return cudaStatus;
	}
    
    //node buffer
    //_cuFinalNodeBufferData = new float* [pxCount];
    cudaStatus = cudaMalloc((void**)&_cuFinalNodeBuffer, sizeof(cuMatrix));
    if (cudaStatus != cudaSuccess)
    {
        makeMallocError("neural network final node array", cudaStatus);
        return cudaStatus;
    }
    
    std::string name = "node matrix array";
    
    cuMatrix temp = cuMatrix(matrix(1, pxCount * _pNet->getShape(_pNet->getShapeLen() - 1), nullptr));
    cudaStatus = cudaMemcpy(_cuFinalNodeBuffer, &temp, sizeof(cuMatrix), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        makeMemcpyError(name.c_str(), cudaStatus);
        return cudaStatus;
    }

    float* tempData;
    size_t tempDataSize = temp.cols * temp.rows * sizeof(float);
    cudaStatus = cudaMalloc((void**)&tempData, tempDataSize);
    if (cudaStatus != cudaSuccess)
    {
        makeMallocError((name + " data").c_str(), cudaStatus);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(tempData, temp.data, tempDataSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        makeMemcpyError((name + " data").c_str(), cudaStatus);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(&(_cuFinalNodeBuffer->data), &tempData, sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        makeMemcpyError((name + " data pointer").c_str(), cudaStatus);
        return cudaStatus;
    }

    _cuFinalNodeBufferData = tempData;
    

    //cuda linker fix prevents anything that uses cuMatrix from linking in c++
    //makeMatrix(_cuFinalNodeBuffer, &_cuFinalNodeBufferData,
    //    cuMatrix(matrix(1, pxCount * _pNet->getShape(_pNet->getShapeLen() - 1), nullptr)), name);

	_grid = dim3(_pWnd->getWidth(), _pWnd->getHeight());

	return cudaStatus;
}

cudaNNetRenderer::cudaNNetRenderer() :
    cudaNNetProcessor(), _pWnd(nullptr),
    _cuFinalNodeBuffer(nullptr), _cuFinalNodeBufferData(nullptr)
{
}

cudaNNetRenderer::~cudaNNetRenderer()
{
	cuSafeFree((void**)&_cuColorBuffer);
    cuSafeFree((void**)&_cuFinalNodeBufferData);
    cuSafeFree((void**)&_cuFinalNodeBuffer);

	//reference pointers
	_pWnd = nullptr;
}

cudaNNetRenderer::cudaNNetRenderer(NNet& pNet, const OGLWindow& wnd) :
    cudaNNetProcessor(pNet), _pWnd(&wnd),
    _cuFinalNodeBuffer(nullptr), _cuFinalNodeBufferData(nullptr)
{
	makeBuffers();
}

cudaNNetRenderer::cudaNNetRenderer(const cudaNNetRenderer& R) :
    cudaNNetProcessor(R), _pWnd(R._pWnd),
    _cuFinalNodeBuffer(nullptr), _cuFinalNodeBufferData(nullptr)
{
	makeBuffers();
}

cudaError cudaNNetRenderer::cudaRenderNNet() const
{
    cudaError (*makeKernelError)(const char*, cudaError) = [](const char* err, cudaError cudaStatus)
    {
        std::cout << "Failure in gpu kernel " << err << "!\n";
        std::cout << "error code : " << cudaGetErrorString(cudaStatus);
        return cudaStatus;
    };

    cudaError cudaStatus = cudaSuccess;
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Failed to synchronize gpu!\n";
        std::cout << "error code : " << cudaGetErrorString(cudaStatus);
        return cudaStatus;
    }

    //render screen
    dim3 threadgrid = dim3(32, 8); // 800*600 is divisible by 256, but 600 is not divisible by 16. 32*8 = 16*16
    dim3 blockgrid = dim3(_grid.x / threadgrid.x, _grid.y / threadgrid.y);

    keForwardProp <<<blockgrid, threadgrid>>> (_cuFinalNodeBuffer, _cuWeights, _cuNumWeights);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        makeKernelError("keForwardProp", cudaStatus);
        return cudaStatus;
    }

    keDisplayNodes <<<blockgrid, threadgrid>>> (_cuColorBuffer, _cuFinalNodeBuffer);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        makeKernelError("keDisplayNodes", cudaStatus);
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

cudaError cudaNNetRenderer::cudaRenderNNetMonolithic() const
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
    dim3 threadgrid = dim3(32, 8); // 800*600 is divisible by 256, but 600 is not divisible by 16. 32*8 = 16*16
    dim3 blockgrid = dim3(_grid.x / threadgrid.x, _grid.y / threadgrid.y);

    kernel <<<blockgrid, threadgrid >>> (_cuColorBuffer, _cuWeights, _cuNumWeights);
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

//Cuda device code
//these decorated functions appear to have to not sit in a class structure.
//Maybe they can be wrapped?

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

//this kernel is slightly faster than its parts and gets to stick around for testing
__global__ void kernel(unsigned char* colorBuffer, cuMatrix* cuWeights, int* cuNumWeights)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * gridDim.x * blockDim.x;

    float X = (((float)x / (float)gridDim.x) * 2.0 - 1.0) * 3.0;
    float Y = (((float)y / (float)gridDim.y) * 2.0 - 1.0) * 3.0;

    float initialNode[2] = { X, Y };
    float result = cuForwardProp(cuMatrix(1, 1, initialNode), cuWeights, cuNumWeights).getData(0, 0);

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

__global__ void keForwardProp(cuMatrix* cuFinalNode, cuMatrix* cuWeights, int* cuNumWeights)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * gridDim.x * blockDim.x;

    float X = (((float)x / (float)(gridDim.x * blockDim.x)) * 2.0 - 1.0) * 3.0;
    float Y = (((float)y / (float)(gridDim.y * blockDim.y)) * 2.0 - 1.0) * 3.0;

    float initialNode[2] = { X, Y };
    cuMatrix output = cuForwardProp(cuMatrix(1, 2, initialNode), cuWeights, cuNumWeights);
    memcpy(&cuFinalNode->data[offset * output.cols * output.rows], output.data, output.cols * output.rows * sizeof(float));
}

__global__ void keDisplayNodes(unsigned char* cuColorBuffer, cuMatrix* cuFinalNode)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * gridDim.x * blockDim.x;

    float result = cuFinalNode->data[offset]; //only valid for a 1d output vector

    result = tanhf(result);
    result = (result + 1.0f) / 2.0f;

    //int icolor = (unsigned char)(result * 255.0f);
    //if (255 < icolor) icolor = 255;
    unsigned char color = (unsigned char)(result * 255.0f);
    unsigned char* px = cuColorBuffer + 4 * offset;
    px[0] = color;
    px[1] = color;
    px[2] = color;
    px[3] = 0xFF;
}

__device__ void setColor(unsigned char* px, int input, int limit)
{
    float S = (float)input / (float)limit;
    S *= 3;
    int is = (int)S;
    S -= (float)is;

    int low = (int)((1.0 - S) * 255.0f);
    int high = (int)(S * 255.0f);

    px[is] = low;
    px[(is + 1) % 3] = high;
}

__global__ void testKernel(unsigned char* colorBuffer, cuMatrix* cuWeights, int* cuNumWeights)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * gridDim.x * blockDim.x;

    int sum = x + y;

    unsigned char color = (unsigned char)(sum * 255 / (800 + 600));
    unsigned char* px = colorBuffer + 4 * offset;
    //px[0] = color;
    //px[1] = color;
    //px[2] = color;
    px[0] = 0x00;
    px[1] = 0x00;
    px[2] = 0x00;
    px[3] = 0xFF;

    setColor(px, threadIdx.x + threadIdx.y * blockDim.x, blockDim.x * blockDim.y);
}

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
}