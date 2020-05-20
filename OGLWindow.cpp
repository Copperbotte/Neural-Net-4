// This simple window renderer came from my Cuda Raytracer project.
//
//CUDA by example
//https://developer.download.nvidia.com/books/cuda-by-example/cuda-by-example-sample.pdf

#include "OGLWindow.h"

#pragma comment (lib, "opengl32.lib")
#pragma comment (lib, "glfw3.lib")
#pragma comment (lib, "glfw3dll.lib")

OGLWindow::OGLWindow()
    : _width(800), _height(600), _name("OGLWindow"), _printFunc(nullptr), _thinkFunc(nullptr), _colorBuffer(nullptr)
{
    _colorBuffer = new unsigned char[4 * _width * _height];
}

OGLWindow::~OGLWindow()
{
    close();
    //delete[] _colorBuffer;
}

OGLWindow::OGLWindow(const char* name)
    : _width(800), _height(600), _name(name), _printFunc(nullptr), _thinkFunc(nullptr), _colorBuffer(nullptr)
{
    _colorBuffer = new unsigned char[4 * _width * _height];
}

OGLWindow::OGLWindow(int width, int height)
    : _width(width), _height(height), _name("OGLWindow"), _printFunc(nullptr), _thinkFunc(nullptr), _colorBuffer(nullptr)
{
    _colorBuffer = new unsigned char[4 * _width * _height];
}

OGLWindow::OGLWindow(const char* name, int width, int height) 
    : _width(width), _height(height), _name(name), _printFunc(nullptr), _thinkFunc(nullptr), _colorBuffer(nullptr)
{
    _colorBuffer = new unsigned char[4 * _width * _height];
}

const char* OGLWindow::getName() const
{
    return _name;
}

void OGLWindow::setName(const char* name)
{
    _name = name;
}

void OGLWindow::setPrintFunc(void (*print)(const char*))
{
    _printFunc = print;
}

void OGLWindow::setThinkFunc(void(*thinkFunc)(OGLWindow* This, double time))
{
    _thinkFunc = thinkFunc;
}

int OGLWindow::getWidth() const
{
    return _width;
}

int OGLWindow::getHeight() const
{
    return _height;
}

GLFWwindow* OGLWindow::getWindowPtr() const
{
    return _window;
}

unsigned char* OGLWindow::getColorBufferPtr() const
{
    return _colorBuffer;
}

void OGLWindow::fillColorBuffer(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
    for (int n = 0; n < _width * _height; ++n)
    {
        char* c = (char*)(_colorBuffer) + 4 * n;
        c[0] = r;
        c[1] = g;
        c[2] = b;
        c[3] = a;
    }
}

void OGLWindow::print(const char* str)
{
    if (_printFunc)
        _printFunc(str);
}

GLFWwindow* OGLWindow::init()
{
    //setPrintFunc([](const char* in){cout << in << '\n'; }); // how to set constant lambda?

    const char* prog_title = "NN4";

    print(prog_title);

    //initialize library
    if (!glfwInit())
    {
        print("GLFW Failed to init");
        _window = nullptr;
        return _window;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    GLFWwindow* window = glfwCreateWindow(_width, _height, prog_title, NULL, NULL);
    if (!window)
    {
        print("GLFW Window failed to create");
        glfwTerminate();
        _window = nullptr;
        return _window;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        print("Failed to initialize GLAD");
        glfwTerminate();
        _window = nullptr;
        return _window;
    }

    print("Opengl");
    print((const char*)glGetString(GL_VERSION));
    print("GLSL");
    print((const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    glViewport(0, 0, _width, _height);

    _window = window;
    return _window;
}

void OGLWindow::think()
{
    if (!_thinkFunc)
        return;

    while (!glfwWindowShouldClose(_window))
    {
        _thinkFunc(this, 0.0);
        glDrawPixels(_width, _height, GL_RGBA, GL_UNSIGNED_BYTE, _colorBuffer);
        glfwSwapBuffers(_window);
        glfwPollEvents();
    }
}

void OGLWindow::close()
{
    if (_colorBuffer)
    {
        delete[] _colorBuffer;
        _colorBuffer = nullptr;
    }

    glfwTerminate();

    print("closed!");
}
