#include <functional>
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

namespace glfw_wrappers {
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
} // namespace glfw_wrappers

static auto supportsGraphics(const VkQueueFamilyProperties &properties)
    -> bool {
  return (properties.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U;
}

static auto
vulkanCount(VkPhysicalDevice device,
            const std::function<void(VkPhysicalDevice, uint32_t *)> &f)
    -> uint32_t {
  uint32_t count = 0;
  f(device, &count);
  return count;
}

static auto queueFamilyPropertiesCount(VkPhysicalDevice device) -> uint32_t {
  return vulkanCount(device, [](VkPhysicalDevice device_, uint32_t *count) {
    vkGetPhysicalDeviceQueueFamilyProperties(device_, count, nullptr);
  });
}

static auto queueFamilyProperties(VkPhysicalDevice device, uint32_t count)
    -> std::vector<VkQueueFamilyProperties> {
  std::vector<VkQueueFamilyProperties> properties(count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, properties.data());
  return properties;
}

static auto graphicsSupportingQueueFamilyIndex(
    const std::vector<VkQueueFamilyProperties> &queueFamilyProperties)
    -> uint32_t {
  return static_cast<uint32_t>(std::distance(
      queueFamilyProperties.begin(),
      std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
                   supportsGraphics)));
}

static auto graphicsSupportingQueueFamilyIndex(VkPhysicalDevice device)
    -> uint32_t {
  return graphicsSupportingQueueFamilyIndex(
      queueFamilyProperties(device, queueFamilyPropertiesCount(device)));
}

static auto presentSupportingQueueFamilyIndex(VkPhysicalDevice device,
                                              VkSurfaceKHR surface)
    -> uint32_t {
  auto index{0U};
  while (index < queueFamilyPropertiesCount(device)) {
    VkBool32 support{0U};
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
    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "Hello Triangle";
    applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    applicationInfo.pEngineName = "No Engine";
    applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    applicationInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &applicationInfo;

    uint32_t extensionCount{0};
    createInfo.ppEnabledExtensionNames =
        glfwGetRequiredInstanceExtensions(&extensionCount);
    createInfo.enabledExtensionCount = extensionCount;

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

    const auto indices{std::set<uint32_t>{
        graphicsSupportingQueueFamilyIndex(physicalDevice),
        presentSupportingQueueFamilyIndex(physicalDevice, surface)}};
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos(indices.size());
    const auto queuePriority{1.0F};
    std::transform(indices.begin(), indices.end(), queueCreateInfos.begin(),
                   [&queuePriority](uint32_t index) {
                     VkDeviceQueueCreateInfo queueCreateInfo{};
                     queueCreateInfo.sType =
                         VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                     queueCreateInfo.queueFamilyIndex = index;
                     queueCreateInfo.queueCount = 1;
                     queueCreateInfo.pQueuePriorities = &queuePriority;
                     return queueCreateInfo;
                   });

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
  auto extensionPropertyCount{
      vulkanCount(device, [](VkPhysicalDevice device_, uint32_t *count) {
        vkEnumerateDeviceExtensionProperties(device_, nullptr, count, nullptr);
      })};

  std::vector<VkExtensionProperties> extensionProperties(
      extensionPropertyCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionPropertyCount,
                                       extensionProperties.data());

  std::vector<std::string> extensionNames(extensionProperties.size());
  std::transform(extensionProperties.begin(), extensionProperties.end(),
                 extensionNames.begin(),
                 [](const VkExtensionProperties &properties) {
                   return properties.extensionName;
                 });

  if (!std::includes(extensionNames.begin(), extensionNames.end(),
                     deviceExtensions.begin(), deviceExtensions.end()))
    return false;

  {
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, nullptr);
    if (count == 0)
      return false;
  }

  {
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, nullptr);
    if (count == 0)
      return false;
  }

  auto index{0U};
  for (const auto properties :
       queueFamilyProperties(device, queueFamilyPropertiesCount(device))) {
    VkBool32 presentSupport{0U};
    vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface,
                                         &presentSupport);
    if (supportsGraphics(properties) && presentSupport != 0U)
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

  VkExtent2D extent = {static_cast<uint32_t>(width),
                       static_cast<uint32_t>(height)};
  extent.width = std::clamp(extent.width, capabilities.minImageExtent.width,
                            capabilities.maxImageExtent.width);
  extent.height = std::clamp(extent.height, capabilities.minImageExtent.height,
                             capabilities.maxImageExtent.height);
  return extent;
}

static void run() {
  glfw_wrappers::Init glfwInitialization;

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  glfw_wrappers::Window glfwWindow{windowWidth, windowHeight};

  vulkan_wrappers::Instance vulkanInstance;
  vulkan_wrappers::Surface vulkanSurface{vulkanInstance.instance,
                                         glfwWindow.window};
  uint32_t deviceCount{0};
  vkEnumeratePhysicalDevices(vulkanInstance.instance, &deviceCount, nullptr);
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(vulkanInstance.instance, &deviceCount,
                             devices.data());

  auto *vulkanPhysicalDevice{suitableDevice(devices, vulkanSurface.surface)};
  vulkan_wrappers::Device vulkanDevice{vulkanPhysicalDevice,
                                       vulkanSurface.surface};

  VkSurfaceCapabilitiesKHR capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      vulkanPhysicalDevice, vulkanSurface.surface, &capabilities);

  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(
      vulkanPhysicalDevice, vulkanSurface.surface, &formatCount, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(vulkanPhysicalDevice,
                                       vulkanSurface.surface, &formatCount,
                                       formats.data());

  uint32_t presentModeCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(
      vulkanPhysicalDevice, vulkanSurface.surface, &presentModeCount, nullptr);
  std::vector<VkPresentModeKHR> presentModes(presentModeCount);
  vkGetPhysicalDeviceSurfacePresentModesKHR(
      vulkanPhysicalDevice, vulkanSurface.surface, &presentModeCount,
      presentModes.data());

  auto surfaceFormat = swapSurfaceFormat(formats);
  auto presentMode = swapPresentMode(presentModes);
  VkExtent2D extent = swapExtent(capabilities, glfwWindow.window);

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

  const std::array<uint32_t, 2> queueFamilyIndices = {
      graphicsSupportingQueueFamilyIndex(vulkanPhysicalDevice),
      presentSupportingQueueFamilyIndex(vulkanPhysicalDevice,
                                        vulkanSurface.surface)};

  if (queueFamilyIndices[0] != queueFamilyIndices[1]) {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
  } else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  createInfo.preTransform = capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;

  createInfo.oldSwapchain = VK_NULL_HANDLE;

  VkSwapchainKHR swapChain = nullptr;
  if (vkCreateSwapchainKHR(vulkanDevice.device, &createInfo, nullptr,
                           &swapChain) != VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain!");
  }

  vkGetSwapchainImagesKHR(vulkanDevice.device, swapChain, &imageCount, nullptr);
  std::vector<VkImage> swapChainImages(imageCount);
  vkGetSwapchainImagesKHR(vulkanDevice.device, swapChain, &imageCount,
                          swapChainImages.data());

  auto swapChainImageFormat = surfaceFormat.format;
  auto swapChainExtent = extent;

  VkQueue graphicsQueue{nullptr};
  vkGetDeviceQueue(vulkanDevice.device,
                   graphicsSupportingQueueFamilyIndex(vulkanPhysicalDevice), 0,
                   &graphicsQueue);
  VkQueue presentQueue{nullptr};
  vkGetDeviceQueue(vulkanDevice.device,
                   presentSupportingQueueFamilyIndex(vulkanPhysicalDevice,
                                                     vulkanSurface.surface),
                   0, &presentQueue);

  while (glfwWindowShouldClose(glfwWindow.window) == 0) {
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
