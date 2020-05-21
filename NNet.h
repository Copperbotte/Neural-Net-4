#pragma once

#include "matrix.h"

class NNet
{
	int _shapelen;
	int* _shape;

	matrix* _weights;

	void BuildData();
	float sigmoid(float x);
	float dsigmoid(float x);

public:
	NNet();
	~NNet();
	NNet(int shapelen, int* shape);
	NNet(NNet &N);

	matrix& getWeights(int n);
	matrix& setWeights(int n, matrix& W);

	NNet& operator= (const NNet& N);

	void randomizeNodes();

	matrix forwardProp(matrix& data);

};

