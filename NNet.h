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
	NNet(const NNet &N);

	matrix& getWeights(int n);
	matrix& setWeights(int n, const matrix& W);

	NNet& operator= (const NNet& N);

	void randomizeNodes();

	void   forwardPropArray(const matrix& data,		  matrix* nodes);
	matrix forwardProp	   (const matrix& data);
	float  backPropArray   (      matrix* data, const matrix* expected,  unsigned int datalen);
	float  backProp		   (const matrix& data, const matrix& expected);
};

