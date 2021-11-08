#ifndef SBASH64_GRAPHICS_GLFW_WRAPPERS_HPP_
#define SBASH64_GRAPHICS_GLFW_WRAPPERS_HPP_

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace glfw_wrappers {
struct Init {
  Init();
  ~Init();

  Init(const Init &) = delete;
  auto operator=(const Init &) -> Init & = delete;
  Init(Init &&) = delete;
  auto operator=(Init &&) -> Init & = delete;
};

struct Window {
  Window(int width, int height);
  ~Window();

  Window(const Window &) = delete;
  auto operator=(const Window &) -> Window & = delete;
  Window(Window &&) = delete;
  auto operator=(Window &&) -> Window & = delete;

  GLFWwindow *window;
};
} // namespace glfw_wrappers

#endif
