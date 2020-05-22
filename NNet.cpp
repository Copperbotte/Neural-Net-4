#include "NNet.h"
#include <Windows.h>
#include <math.h>
#include <string>
#include "rand.h"

void NNet::BuildData()
{
	_weights = new matrix[_shapelen - 1];
	for (int n = 0; n < _shapelen - 1; ++n)
		_weights[n] = matrix(_shape[n] + 1, _shape[n + 1], nullptr);
	//[n] + 1 is the bias
}

float NNet::sigmoid(float x)
{
	return tanh(x);
}

float NNet::dsigmoid(float x)
{
	float sig = sigmoid(x);
	return 1.0 - sig * sig;
}

NNet::NNet():_shapelen(5)
{
	int shape[5] = { 2, 3, 5, 3, 1 };
	_shape = new int[5];
	memcpy(_shape, shape, _shapelen * sizeof(int));
	BuildData();
}

NNet::~NNet()
{
	delete[] _weights;
	delete[] _shape;
	_weights = nullptr;
	_shape = nullptr;
}

NNet::NNet(int shapelen, int* shape):_shapelen(shapelen)
{
	_shape = new int[_shapelen];
	memcpy(_shape, shape, shapelen * sizeof(int));
	BuildData();
}

NNet::NNet(const NNet& N):_shapelen(N._shapelen)
{
	_shape = new int[_shapelen];
	memcpy(_shape, N._shape, _shapelen * sizeof(int));
	BuildData();
	for (int n = 0; n < _shapelen - 1; ++n)
		_weights[n] = N._weights[n];
}

matrix& NNet::getWeights(int n)
{
	bool e_range = n < 0 || _shapelen - 2 < n;
	if (e_range)
	{
		std::string e_message("Neural Net weight access out of range! ");
		e_message += "Nodes: " + std::to_string(_shapelen) + ", Accessed: " + std::to_string(n) + "\n";
		throw std::out_of_range(e_message);
		return _weights[0];
	}

	return _weights[n];
}

matrix& NNet::setWeights(int n, const matrix& W)
{
	bool e_range = n < 0 || _shapelen - 2 < n;
	if (e_range)
	{
		std::string e_message("Neural Net weight access out of range! ");
		e_message += "Nodes: " + std::to_string(_shapelen) + ", Accessed: " + std::to_string(n) + "\n";
		throw std::out_of_range(e_message);
		return _weights[0];
	}

	_weights[n] = W;

	return _weights[n];
}

NNet& NNet::operator=(const NNet& N)
{
	if (_weights) delete[] _weights;
	if (_shape) delete[] _shape;
	_shapelen = N._shapelen;
	_shape = new int[N._shapelen];
	memcpy(_shape, N._shape, _shapelen * sizeof(int));
	BuildData();
	for (int n = 0; n < _shapelen - 1; ++n)
		_weights[n] = N._weights[n];
	return *this;
}

void NNet::randomizeNodes()
{
	random Random = random();
	for (int n = 0; n < _shapelen - 1; ++n)
		for (int c = 0; c < _weights[n].getCols(); ++c)
			for (int r = 0; r < _weights[n].getRows(); ++r)
				_weights[n].setData(c, r, Random.xorshfdbl());
}

void NNet::forwardPropArray(const matrix& data, matrix* nodes)
{
	//copy matrix, with an extra slot as a bias
	nodes[0] = data;
	for (int i = 0; i < _shapelen - 1; ++i)
	{
		matrix bias = matrix(1, nodes[i].getRows() + 1, nullptr);
		for (int r = 0; r < nodes[i].getRows(); ++r)
			bias.setData(0, r + 1, nodes[i].getData(0, r));
		bias.setData(0, 0, 1.0);

		//sigmoid input
		for (int r = 1; r < bias.getRows(); ++r)
			bias.setData(0, r, sigmoid(bias.getData(0, r)));

		//forward propogate
		nodes[i+1] = _weights[i] * bias;
	}
}

matrix NNet::forwardProp(const matrix& data)
{
	//copy matrix, with an extra slot as a bias
	matrix node = data;
	for (int i = 0; i < _shapelen - 1; ++i)
	{
		matrix bias = matrix(1, node.getRows() + 1, nullptr);
		for (int r = 0; r < node.getRows(); ++r)
			bias.setData(0, r+1, node.getData(0, r));
		bias.setData(0, 0, 1.0);

		//sigmoid input
		for (int r = 1; r < bias.getRows(); ++r)
			bias.setData(0, r, sigmoid(bias.getData(0, r)));

		//forward propogate
		node = _weights[i] * bias;
	}
	
	return node;
}

float NNet::backPropArray(matrix* data, const matrix* expected, unsigned int datalen)
{
	//build node array
	matrix* nodes = new matrix[_shapelen];

	matrix* _weightDelta = new matrix[_shapelen - 1];
	for (int n = 0; n < _shapelen - 1; ++n)
		_weightDelta[n] = matrix(_shape[n] + 1, _shape[n + 1], nullptr);
	
	float batchError = 0.0f;

	for (int i = 0; i < datalen; ++i)
	{
		forwardPropArray(data[i], nodes);

		matrix deltaNode = nodes[_shapelen - 1] - expected[i];
		float error = (deltaNode.transpose() * deltaNode).getData(0, 0) / 4.0;
		batchError += error;

		for (int n = _shapelen - 2; 0 <= n; --n)
		{
			//add bias
			matrix sig(1, nodes[n].getRows() + 1, nullptr);
			sig.setData(0, 0, 1.0);
			for (int r = 0; r < nodes[n].getRows(); ++r)
				sig.setData(0, r + 1, nodes[n].getData(0, r));

			matrix dsig = sig;

			for (int r = 1; r < sig.getRows(); ++r)
			{
				sig.setData(0, r, sigmoid(sig.getData(0, r)));
				dsig.setData(0, r, dsigmoid(dsig.getData(0, r)));
			}

			deltaNode = deltaNode.transpose();

			//delta NNet is the unique combo of sigma(N1) * dN2
			_weightDelta[n] += (sig * deltaNode).transpose();

			//backpropogate to the previous node
			matrix dNodeBias = (deltaNode * _weights[n]).transpose();

			//strip bias
			deltaNode = matrix(1, dNodeBias.getRows() - 1, nullptr);
			for (int r = 0; r < deltaNode.getRows(); ++r)
				deltaNode.setData(0, r, dNodeBias.getData(0, r + 1));

			//sigmoid derivative
			for (int r = 0; r < deltaNode.getRows(); ++r)
				deltaNode.setData(0, r, deltaNode.getData(0, r) * 0.5 * dsig.getData(0, r + 1));
		}
	}

	float rate = 0.5;
	for (int n = 0; n < _shapelen - 1; ++n)
		_weights[n] -= _weightDelta[n] * (rate / (float)datalen);

	delete[] nodes;
	delete[] _weightDelta;

	return batchError / (float)datalen;
}

float NNet::backProp(const matrix& data, const matrix& expected)
{
	//build node array
	matrix* nodes = new matrix[_shapelen];
	forwardPropArray(data, nodes);
	
	matrix* _weightDelta = new matrix[_shapelen - 1];
	for (int n = 0; n < _shapelen - 1; ++n)
		_weightDelta[n] = matrix(_shape[n] + 1, _shape[n + 1], nullptr);

	matrix deltaNode = nodes[_shapelen - 1] - expected;
	float error = (deltaNode.transpose() * deltaNode).getData(0, 0) / 4.0;

	for (int n = _shapelen - 2; 0 <= n; --n)
	{
		//add bias
		matrix sig(1, nodes[n].getRows() + 1, nullptr);
		sig.setData(0, 0, 1.0);
		for (int r = 0; r < nodes[n].getRows(); ++r)
			sig.setData(0, r + 1, nodes[n].getData(0, r));

		matrix dsig = sig;

		for (int r = 1; r < sig.getRows(); ++r)
		{
			sig.setData(0, r, sigmoid(sig.getData(0, r)));
			dsig.setData(0, r, dsigmoid(dsig.getData(0, r)));
		}
		
		deltaNode = deltaNode.transpose();

		//delta NNet is the unique combo of sigma(N1) * dN2
		_weightDelta[n] = (sig * deltaNode).transpose();
		
		//backpropogate to the previous node
		matrix dNodeBias = (deltaNode * _weights[n]).transpose();

		//strip bias
		deltaNode = matrix(1, dNodeBias.getRows() - 1, nullptr);
		for (int r = 0; r < deltaNode.getRows(); ++r)
			deltaNode.setData(0, r, dNodeBias.getData(0, r + 1));

		//sigmoid derivative
		for (int r = 0; r < deltaNode.getRows(); ++r)
			deltaNode.setData(0, r, deltaNode.getData(0, r) * 0.5 * dsig.getData(0, r + 1));
	}

	float rate = 0.5;
	for (int n = 0; n < _shapelen - 1; ++n)
		_weights[n] -= _weightDelta[n] * rate;

	delete[] nodes;
	delete[] _weightDelta;

	return error;
}


