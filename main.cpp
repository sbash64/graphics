#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iterator>
#include <set>
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

static auto graphicsSupportIndex(VkPhysicalDevice device) -> uint32_t {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilies.data());

  return static_cast<uint32_t>(
      std::distance(queueFamilies.begin(),
                    std::find_if(queueFamilies.begin(), queueFamilies.end(),
                                 [](const VkQueueFamilyProperties &properties) {
                                   return (properties.queueFlags &
                                           VK_QUEUE_GRAPHICS_BIT) != 0U;
                                 })));
}

static auto presentSupportIndex(VkPhysicalDevice device, VkSurfaceKHR surface)
    -> uint32_t {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  auto index{0U};
  while (index < queueFamilyCount) {
    VkBool32 presentSupport = 0U;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface,
                                         &presentSupport);
    if (presentSupport != 0U)
      return index;
    ++index;
  }
  throw std::runtime_error{"no present support index found"};
}

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
    createInfo.pNext = nullptr;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
      throw std::runtime_error("failed to create instance!");
  }

  ~Instance() { vkDestroyInstance(instance, nullptr); }

  VkInstance instance{};
};

struct Device {
  explicit Device(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

    for (auto queueFamily :
         std::set<uint32_t>{graphicsSupportIndex(physicalDevice),
                            presentSupportIndex(physicalDevice, surface)}) {
      VkDeviceQueueCreateInfo queueCreateInfo{};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      auto queuePriority{1.0F};
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount = 0;

    createInfo.enabledLayerCount = 0;

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create logical device!");
  }

  ~Device() { vkDestroyDevice(device, nullptr); }

  VkDevice device{};
};

struct Surface {
  Surface(VkInstance instance, GLFWwindow *window) : instance{instance} {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create window surface!");
  }

  ~Surface() { vkDestroySurfaceKHR(instance, surface, nullptr); }

  VkInstance instance;
  VkSurfaceKHR surface{};
};
} // namespace vulkan_wrappers

constexpr auto windowWidth = 800;
constexpr auto windowHeight = 600;

static auto suitable(VkPhysicalDevice device, VkSurfaceKHR surface) -> bool {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilies.data());

  auto index{0U};
  for (auto properties : queueFamilies) {
    VkBool32 presentSupport = 0U;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface,
                                         &presentSupport);
    if ((properties.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U &&
        (presentSupport != 0U))
      return true;
    ++index;
  }
  return false;
}

static auto suitableDevice(const std::vector<VkPhysicalDevice> &devices,
                           VkSurfaceKHR surface) -> VkPhysicalDevice {
  for (const auto &device : devices)
    if (suitable(device, surface))
      return device;
  throw std::runtime_error("failed to find a suitable GPU!");
}

static void run() {
  gflw_wrappers::Init gflwInitialization;

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  gflw_wrappers::Window gflwWindow{windowWidth, windowHeight};

  vulkan_wrappers::Instance vulkanInstance;
  vulkan_wrappers::Surface vulkanSurface{vulkanInstance.instance,
                                         gflwWindow.window};
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(vulkanInstance.instance, &deviceCount, nullptr);

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(vulkanInstance.instance, &deviceCount,
                             devices.data());

  auto *physicalDevice{suitableDevice(devices, vulkanSurface.surface)};
  vulkan_wrappers::Device device{physicalDevice, vulkanSurface.surface};
  VkQueue graphicsQueue = nullptr;
  vkGetDeviceQueue(device.device, graphicsSupportIndex(physicalDevice), 0,
                   &graphicsQueue);
  VkQueue presentQueue = nullptr;
  vkGetDeviceQueue(device.device,
                   presentSupportIndex(physicalDevice, vulkanSurface.surface),
                   0, &presentQueue);

  while (glfwWindowShouldClose(gflwWindow.window) == 0) {
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
