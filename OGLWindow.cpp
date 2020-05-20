// This simple window renderer came from my Cuda Raytracer project.
//
//CUDA by example
//https://developer.download.nvidia.com/books/cuda-by-example/cuda-by-example-sample.pdf

#include <iostream>
#include <Windows.h>

#include <fstream>
#include <string>
#include <sstream>
#include <random>

#include "glad/glad.h"
#include "glfw/glfw3.h"

#pragma comment (lib, "opengl32.lib")
#pragma comment (lib, "glfw3.lib")
#pragma comment (lib, "glfw3dll.lib")

using namespace std;

const int width = 800,
height = 600;
unsigned char colorbuffer[width * height * 4];

int clamp(int x, int low, int high)
{
    if (x < low) x = low;
    if (high < x) x = high;
    return x;
}

int oglWindow()
{
    const char* prog_title = "NN4";
    cout << prog_title << '\n';

    //initialize library
    if (!glfwInit())
    {
        cout << "GLFW Failed to init" << endl;
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    GLFWwindow* window = glfwCreateWindow(width, height, prog_title, NULL, NULL);
    if (!window)
    {
        cout << "GLFW Window failed to create" << endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        cout << "Failed to initialize GLAD" << endl;
        glfwTerminate();
        return -1;
    }

    cout << "Opengl " << glGetString(GL_VERSION) << " GLSL" << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

    glViewport(0, 0, width, height);

    for (int i = 0; i < width * height * 4; ++i)
        colorbuffer[i] = 0xFF;


    for (int n = 0; n < width * height; ++n)
    {
        char* c = (char*)(colorbuffer)+4 * n;
        c[0] = 0xFF;
        c[1] = 0x8F;
        c[2] = 0x00;
    }

    while (!glfwWindowShouldClose(window))
    {
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, colorbuffer);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}
