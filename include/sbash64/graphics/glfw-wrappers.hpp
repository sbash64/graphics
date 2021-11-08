
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace glfw_wrappers {
struct Init {
  Init() { glfwInit(); }

  ~Init() { glfwTerminate(); }

  Init(const Init &) = delete;
  auto operator=(const Init &) -> Init & = delete;
  Init(Init &&) = delete;
  auto operator=(Init &&) -> Init & = delete;
};

struct Window {
  Window(int width, int height)
      : window{glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr)} {}

  ~Window() { glfwDestroyWindow(window); }

  Window(const Window &) = delete;
  auto operator=(const Window &) -> Window & = delete;
  Window(Window &&) = delete;
  auto operator=(Window &&) -> Window & = delete;

  GLFWwindow *window;
};
} // namespace glfw_wrappers