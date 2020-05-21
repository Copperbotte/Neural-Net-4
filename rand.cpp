
#include <Windows.h>
#include "rand.h"


random::random()
{
	unsigned long init[3] = { 123456789, 362436069, 521288629 };
	memcpy(xorshfnums, &init, sizeof(init));
}

random::~random()
{
}

unsigned long random::xorshf96(void) // http://stackoverflow.com/questions/1640258/need-a-fast-random-generator-for-c
{          //period 2^96-1
	unsigned long t;
	xorshfnums[0] ^= xorshfnums[0] << 16;
	xorshfnums[0] ^= xorshfnums[0] >> 5;
	xorshfnums[0] ^= xorshfnums[0] << 1;

	t = xorshfnums[0];
	xorshfnums[0] = xorshfnums[1];
	xorshfnums[1] = xorshfnums[2];
	xorshfnums[2] = t ^ xorshfnums[0] ^ xorshfnums[1];

	return xorshfnums[2];
}
double random::xorshfdbl(void)
{ // Double: sign bit, 11 exponent bits, 52 fraction bits,  0x3ff0000000000000 = Exponent and Power section, equivelant to 1
	unsigned long long x = 0x3ff0000000000000 | ((unsigned long long)xorshf96() << 20); //xorshft92 is 32 bits long, 32 - 52 = 20 bits shifted
	return *(double*)&x - 1.0;
}
unsigned long* random::xorshfdata(void)
{
	return xorshfnums;
}