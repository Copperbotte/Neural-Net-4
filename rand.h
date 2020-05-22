#pragma once

class random
{
	unsigned long xorshfnums[3];

public:
	random(unsigned long seed);
	~random();

	unsigned long xorshf96(void);
	double xorshfdbl(void);
	unsigned long* xorshfdata(void);
};


