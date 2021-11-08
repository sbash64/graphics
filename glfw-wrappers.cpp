#include <sbash64/graphics/glfw-wrappers.hpp>

namespace glfw_wrappers {
Init::Init() { glfwInit(); }

Init::~Init() { glfwTerminate(); }

Window::Window(int width, int height)
    : window{glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr)} {}

Window::~Window() { glfwDestroyWindow(window); }
} // namespace glfw_wrappers
