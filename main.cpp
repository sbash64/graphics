#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
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
  uint32_t queueFamilyPropertyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyPropertyCount,
                                           nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilyProperties(
      queueFamilyPropertyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyPropertyCount,
                                           queueFamilyProperties.data());

  return static_cast<uint32_t>(std::distance(
      queueFamilyProperties.begin(),
      std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
                   [](const VkQueueFamilyProperties &properties) {
                     return (properties.queueFlags & VK_QUEUE_GRAPHICS_BIT) !=
                            0U;
                   })));
}

static auto presentSupportIndex(VkPhysicalDevice device, VkSurfaceKHR surface)
    -> uint32_t {
  uint32_t queueFamilyPropertyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyPropertyCount,
                                           nullptr);

  auto index{0U};
  while (index < queueFamilyPropertyCount) {
    VkBool32 support = 0U;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &support);
    if (support != 0U)
      return index;
    ++index;
  }
  throw std::runtime_error{"no present support index found"};
}

constexpr std::array<const char *, 1> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

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

    const auto queuePriority{1.0F};
    for (auto queueFamily :
         std::set<uint32_t>{graphicsSupportIndex(physicalDevice),
                            presentSupportIndex(physicalDevice, surface)}) {
      VkDeviceQueueCreateInfo queueCreateInfo{};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
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
    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
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

constexpr auto windowWidth{800};
constexpr auto windowHeight{600};

static auto suitable(VkPhysicalDevice device, VkSurfaceKHR surface) -> bool {
  uint32_t queueFamilyPropertyCount{0};
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyPropertyCount,
                                           nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilyProperties(
      queueFamilyPropertyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyPropertyCount,
                                           queueFamilyProperties.data());

  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       nullptr);

  std::vector<VkExtensionProperties> extensionProperties(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       extensionProperties.data());

  std::set<std::string> requiredExtensions{deviceExtensions.begin(),
                                           deviceExtensions.end()};

  for (const auto &extension : extensionProperties) {
    requiredExtensions.erase(extension.extensionName);
  }

  if (!requiredExtensions.empty())
    return false;

  uint32_t presentModeCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount,
                                            nullptr);
  if (presentModeCount == 0)
    return false;

  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
  if (formatCount == 0)
    return false;

  auto index{0U};
  for (auto properties : queueFamilyProperties) {
    VkBool32 presentSupport{0U};
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

static auto swapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats)
    -> VkSurfaceFormatKHR {
  for (const auto &format : formats)
    if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
      return format;
  return formats[0];
}

static auto swapPresentMode(const std::vector<VkPresentModeKHR> &presentModes)
    -> VkPresentModeKHR {
  for (const auto &mode : presentModes)
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
      return mode;
  return VK_PRESENT_MODE_FIFO_KHR;
}

static auto swapExtent(VkSurfaceCapabilitiesKHR capabilities,
                       GLFWwindow *window) -> VkExtent2D {
  if (capabilities.currentExtent.width != UINT32_MAX)
    return capabilities.currentExtent;
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(window, &width, &height);

  VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                             static_cast<uint32_t>(height)};

  actualExtent.width =
      std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                 capabilities.maxImageExtent.width);
  actualExtent.height =
      std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                 capabilities.maxImageExtent.height);

  return actualExtent;
}

static void run() {
  gflw_wrappers::Init gflwInitialization;

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  gflw_wrappers::Window gflwWindow{windowWidth, windowHeight};

  vulkan_wrappers::Instance vulkanInstance;
  vulkan_wrappers::Surface vulkanSurface{vulkanInstance.instance,
                                         gflwWindow.window};
  uint32_t deviceCount{0};
  vkEnumeratePhysicalDevices(vulkanInstance.instance, &deviceCount, nullptr);

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(vulkanInstance.instance, &deviceCount,
                             devices.data());

  auto *physicalDevice{suitableDevice(devices, vulkanSurface.surface)};
  vulkan_wrappers::Device device{physicalDevice, vulkanSurface.surface};

  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
  {
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        physicalDevice, vulkanSurface.surface, &capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, vulkanSurface.surface,
                                         &formatCount, nullptr);

    if (formatCount != 0) {
      formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(
          physicalDevice, vulkanSurface.surface, &formatCount, formats.data());
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        physicalDevice, vulkanSurface.surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
      presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(
          physicalDevice, vulkanSurface.surface, &presentModeCount,
          presentModes.data());
    }
  }

  auto surfaceFormat = swapSurfaceFormat(formats);
  auto presentMode = swapPresentMode(presentModes);
  VkExtent2D extent = swapExtent(capabilities, gflwWindow.window);

  uint32_t imageCount = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0 &&
      imageCount > capabilities.maxImageCount) {
    imageCount = capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = vulkanSurface.surface;

  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  uint32_t queueFamilyIndices[] = {
      graphicsSupportIndex(physicalDevice),
      presentSupportIndex(physicalDevice, vulkanSurface.surface)};

  if (queueFamilyIndices[0] != queueFamilyIndices[1]) {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  createInfo.preTransform = capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;

  createInfo.oldSwapchain = VK_NULL_HANDLE;

  VkSwapchainKHR swapChain = nullptr;
  if (vkCreateSwapchainKHR(device.device, &createInfo, nullptr, &swapChain) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain!");
  }

  vkGetSwapchainImagesKHR(device.device, swapChain, &imageCount, nullptr);
  std::vector<VkImage> swapChainImages(imageCount);
  vkGetSwapchainImagesKHR(device.device, swapChain, &imageCount,
                          swapChainImages.data());

  auto swapChainImageFormat = surfaceFormat.format;
  auto swapChainExtent = extent;

  VkQueue graphicsQueue{nullptr};
  vkGetDeviceQueue(device.device, graphicsSupportIndex(physicalDevice), 0,
                   &graphicsQueue);
  VkQueue presentQueue{nullptr};
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
