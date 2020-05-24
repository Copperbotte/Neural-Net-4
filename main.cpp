
#include <iostream>
#include <Windows.h>
#include "cudaNNetProcessor.h"
#include "OGLWindow.h"
#include "NNet.h"
#include "rand.h"

using namespace std;

void generateSampleFunctionData(matrix** i_matrix, matrix** o_matrix, float (*test_func)(float x, float y));

int main()
{
    //int shape[] = { 2, 3, 1 };
    int shape[] = { 2, 1 };
    int shapelen = sizeof(shape) / sizeof(int);

    //NNet net = NNet();
    NNet net = NNet(shapelen, shape);
    net.randomizeNodes(GetTickCount());
    net.setRate(0.25);

    float (*test_func)(float, float) = [](float x, float y)
    {
        //float X = x - 1;
        //if (X * X + y * y < 4.0) return 1.0f;
        //return -1.0f;
        //return sin(2.0f * x) * sin(2.0f * y);
        return x + y - 2;
    };

    matrix* i_matrix;
    matrix* o_matrix;
    generateSampleFunctionData(&i_matrix, &o_matrix, test_func);

    OGLWindow wnd("NN4");
    wnd.setPrintFunc([](const char* str) {cout << str << '\n'; });
    wnd.fillColorBuffer(0xFF, 0x8F, 0x00, 0xFF);
    wnd.init();

    cudaNNetProcessor cnnp(net, wnd);
    cudaError cudaStatus = cudaSuccess;

    while (wnd.thinkStep())
    {
        cudaStatus = cnnp.cudaCopyNNet();
        cudaStatus = cnnp.cudaRenderNNet();

        float err = 0.0f;
        for (int i = 0; i < 100; ++i)
            err = net.backPropArray(i_matrix, o_matrix, 100);
        cout << err << '\n';
    }

    delete[] i_matrix;
    delete[] o_matrix;

    return 0;
}

void generateSampleFunctionData(matrix** i_matrix, matrix** o_matrix, float (*test_func)(float x, float y))
{
    random Random(GetTickCount() + 12345);

    float** i_function = new float* [100];
    float* o_function = new float[100];
    *i_matrix = new matrix[100];
    *o_matrix = new matrix[100];

    for (int i = 0; i < 100; ++i)
    {
        i_function[i] = new float[2];
        float x = Random.xorshfdbl() * 6.0 - 3.0;
        float y = Random.xorshfdbl() * 6.0 - 3.0;
        float o = test_func(x, y);
        i_function[i][0] = x;
        i_function[i][1] = y;
        o_function[i] = o;
        (*i_matrix)[i] = matrix(1, 2, i_function[i]);
        (*o_matrix)[i] = matrix(1, 1, nullptr);
        (*o_matrix)[i].setData(0, 0, o_function[i]);
    }

    for (int i = 0; i < 100; ++i)
        delete[] i_function[i];
    delete[] i_function;
    delete[] o_function;

}