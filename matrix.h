#pragma once
#include <Windows.h>
#include <stdexcept>

#include <Windows.h>
#include <stdexcept>

class matrix
{
	int cols, rows;
	float* data;

public:
	matrix();
	~matrix();
	matrix(const matrix& copy);
	matrix(int cols, int rows, const float* data);

	static matrix identity(int cols, int rows);
	static matrix identity(int n);

	//helper functions
	int   getRows()	const;
	int   getCols() const;
	const float* getDataPtr() const;
	float getData(int c, int r) const;
	float setData(int c, int r, float d);

	//unary operators
	matrix& operator= (const matrix& R);
	matrix  operator- (void) const;
	matrix& operator+=(const matrix& R);
	matrix& operator-=(const matrix& R);
	matrix& operator*=(float   f);
	matrix& operator*=(const matrix& R);
	matrix& operator/=(float   f);

	//binary operators
	matrix  operator+ (const matrix& R) const;
	matrix  operator- (const matrix& R) const;
	matrix  operator* (const matrix& R) const;

	//transform operations
	matrix  transpose() const;
	matrix  inverse()   const;
};



