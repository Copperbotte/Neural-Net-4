
#include "cudaNNetRenderer.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

//Cuda requires(?) all device code to be in the same file. This device code currently sits at the bottom of this file.
__global__ void kernel(unsigned char* colorBuffer, cuMatrix* cuWeights, int* cuNumWeights);

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


	//color buffer
	cudaStatus = cudaMalloc((void**)&_cuColorBuffer, _pWnd->getWidth() * _pWnd->getHeight() * 4 * sizeof(char));
	if (cudaStatus != cudaSuccess)
	{
		makeMallocError("bitmap", cudaStatus);
		return cudaStatus;
	}

	_grid = dim3(_pWnd->getWidth(), _pWnd->getHeight());

	return cudaStatus;
}

cudaNNetRenderer::cudaNNetRenderer() : cudaNNetProcessor(), _pWnd(nullptr)
{
}

cudaNNetRenderer::~cudaNNetRenderer()
{
	safeFree((void**)&_cuColorBuffer);

	//reference pointers
	_pWnd = nullptr;
}

cudaNNetRenderer::cudaNNetRenderer(NNet& pNet, const OGLWindow& wnd) : cudaNNetProcessor(pNet), _pWnd(&wnd)
{
	makeBuffers();
}

cudaNNetRenderer::cudaNNetRenderer(const cudaNNetRenderer& R) : cudaNNetProcessor(R), _pWnd(R._pWnd)
{
	makeBuffers();
}

cudaError cudaNNetRenderer::cudaRenderNNet() const
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

//Cuda device code
//these decorated functions appear to have to not sit in a class structure.
//Maybe they can be wrapped?

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

__global__ void kernel(unsigned char* colorBuffer, cuMatrix* cuWeights, int* cuNumWeights)
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