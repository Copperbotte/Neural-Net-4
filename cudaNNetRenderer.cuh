#pragma once

#include "cudaNNetProcessor.cuh"

class cudaNNetRenderer : public cudaNNetProcessor
{
private:
	unsigned char* _cuColorBuffer;

	const OGLWindow* _pWnd;

	dim3 _grid;

	cudaError makeBuffers();

public:

	cudaNNetRenderer();
	~cudaNNetRenderer();
	cudaNNetRenderer(NNet& pNet, const OGLWindow& wnd);
	cudaNNetRenderer(const cudaNNetRenderer& R);

	cudaError cudaRenderNNet() const;
};