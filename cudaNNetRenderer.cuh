#pragma once

#include "cudaNNetProcessor.cuh"

class cudaNNetRenderer : public cudaNNetProcessor
{
private:
	unsigned char* _cuColorBuffer;
	cuMatrix* _cuFinalNodeBuffer;
	float* _cuFinalNodeBufferData;

	const OGLWindow* _pWnd;

	dim3 _grid;

	cudaError makeBuffers();

public:

	cudaNNetRenderer();
	~cudaNNetRenderer();
	cudaNNetRenderer(NNet& pNet, const OGLWindow& wnd);
	cudaNNetRenderer(const cudaNNetRenderer& R);

	cudaError cudaRenderNNet() const;
	cudaError cudaNNetRenderer::cudaRenderNNetMonolithic() const;
};