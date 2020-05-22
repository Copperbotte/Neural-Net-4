#pragma once

#include "glad/glad.h"
#include "glfw/glfw3.h"

class OGLWindow
{
	const int _width;
	const int _height;
	const char* _name;
	void (*_printFunc)(const char*);
	void (*_thinkFunc)(OGLWindow* This, double time);
	unsigned char* _colorBuffer;
	GLFWwindow* _window;

public:
	OGLWindow();
	~OGLWindow();
	OGLWindow(const char* name);
	OGLWindow(int width, int height);
	OGLWindow(const char* name, int width, int height);

	const char* getName() const;
	void setName(const char* name);
	//getPrintFunc(); //How to return a function pointer?
	void setPrintFunc(void (*print)(const char*));
	void setThinkFunc(void (*thinkFunc)(OGLWindow* This, double time));

	int getWidth() const;
	int getHeight() const;
	GLFWwindow* getWindowPtr() const;

	unsigned char* getColorBufferPtr() const;
	void fillColorBuffer(unsigned char r, unsigned char g,
		unsigned char b, unsigned char a);

	void print(const char* str);

	GLFWwindow* init();
	void think();
	bool thinkStep();

	void close();

};