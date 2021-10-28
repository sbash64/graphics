#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan_core.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace gflw_wrappers {
struct Init {
  Init() { glfwInit(); }

  ~Init() { glfwTerminate(); }
};

struct Window {
  Window(int width, int height)
      : window{glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr)} {}

  ~Window() { glfwDestroyWindow(window); }

  GLFWwindow *window;
};
} // namespace gflw_wrappers

namespace vulkan_wrappers {
struct Instance {
  Instance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    uint32_t glfwExtensionCount = 0;
    createInfo.ppEnabledExtensionNames =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    createInfo.enabledExtensionCount = glfwExtensionCount;

    createInfo.enabledLayerCount = 0;
    createInfo.enabledLayerCount = 0;

    createInfo.pNext = nullptr;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
      throw std::runtime_error("failed to create instance!");
  }

  ~Instance() { vkDestroyInstance(instance, nullptr); }

  VkInstance instance{};
};
} // namespace vulkan_wrappers

constexpr auto windowWidth = 800;
constexpr auto windowHeight = 600;

static void run() {
  gflw_wrappers::Init gflwInitialization;

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  gflw_wrappers::Window window{windowWidth, windowHeight};

  vulkan_wrappers::Instance vulkanInstance;
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(vulkanInstance.instance, &deviceCount, nullptr);

  if (deviceCount == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(vulkanInstance.instance, &deviceCount,
                             devices.data());

  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  for (const auto &device : devices) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());

    for (const auto &queueFamily : queueFamilies) {
      if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U) {
        physicalDevice = device;
        break;
      }
    }
    if (physicalDevice != VK_NULL_HANDLE)
      break;
  }

  if (physicalDevice == VK_NULL_HANDLE) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }

  while (glfwWindowShouldClose(window.window) == 0) {
    glfwPollEvents();
  }
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
