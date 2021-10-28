#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <exception>
#include <iostream>

namespace gflw_wrappers {
struct Init {
  Init() { glfwInit(); }

  ~Init() { glfwTerminate(); }
};
} // namespace gflw_wrappers

constexpr auto windowWidth = 800;
constexpr auto windowHeight = 600;

static void run() {
  gflw_wrappers::Init gflwInitialization;

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  auto *window =
      glfwCreateWindow(windowWidth, windowHeight, "Vulkan", nullptr, nullptr);
  while (glfwWindowShouldClose(window) == 0) {
    glfwPollEvents();
  }
  glfwDestroyWindow(window);
}

int main() {
  try {
    run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
