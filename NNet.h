#pragma once

#include "matrix.h"

class NNet
{
	int _shapelen;

	int* _shape;
	matrix* _weights;

	typedef float (*p_sFunc)(float);
	p_sFunc*  _sigmoids;
	p_sFunc* _dsigmoids;

	float _rate;

	void BuildData();
	static float s_tanh(float x);
	static float s_dtanh(float x);
	static float s_const(float x);
	static float s_dconst(float x);

public:
	NNet();
	~NNet();
	NNet(int shapelen, int* shape);
	NNet(const NNet &N);

	NNet& operator= (const NNet& N);

	int getShapeLen() const;
	int getShape(int n) const;

	const matrix& getWeights(int n) const;
		  matrix& setWeights(int n, const matrix& W);
	
	void  setSigmoid(int n, p_sFunc func);
	void setdSigmoid(int n, p_sFunc func);

	float getRate() const;
	float setRate(float rate);

	void randomizeNodes(unsigned long seed);

	void   forwardPropArray(const matrix& data,		  matrix* nodes);
	matrix forwardProp	   (const matrix& data);
	float  backPropArray   (      matrix* data, const matrix* expected,  unsigned int datalen);
	float  backProp		   (const matrix& data, const matrix& expected);
};

