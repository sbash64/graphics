#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan_core.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

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

static auto supportsGraphics(const VkQueueFamilyProperties &properties)
    -> bool {
  return (properties.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U;
}

static auto vulkanCountFromPhysicalDevice(
    VkPhysicalDevice device,
    const std::function<void(VkPhysicalDevice, uint32_t *)> &f) -> uint32_t {
  uint32_t count{0};
  f(device, &count);
  return count;
}

static auto queueFamilyPropertiesCount(VkPhysicalDevice device) -> uint32_t {
  return vulkanCountFromPhysicalDevice(
      device, [](VkPhysicalDevice device_, uint32_t *count) {
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
  uint32_t index{0U};
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

static auto swapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats)
    -> VkSurfaceFormatKHR {
  for (const auto &format : formats)
    if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
      return format;
  return formats.front();
}

static auto swapPresentMode(const std::vector<VkPresentModeKHR> &presentModes)
    -> VkPresentModeKHR {
  for (const auto &mode : presentModes)
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
      return mode;
  return VK_PRESENT_MODE_FIFO_KHR;
}

static auto swapExtent(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
                       GLFWwindow *window) -> VkExtent2D {
  VkSurfaceCapabilitiesKHR capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface,
                                            &capabilities);

  if (capabilities.currentExtent.width != UINT32_MAX)
    return capabilities.currentExtent;

  int width{0};
  int height{0};
  glfwGetFramebufferSize(window, &width, &height);

  VkExtent2D extent{static_cast<uint32_t>(width),
                    static_cast<uint32_t>(height)};
  extent.width = std::clamp(extent.width, capabilities.minImageExtent.width,
                            capabilities.maxImageExtent.width);
  extent.height = std::clamp(extent.height, capabilities.minImageExtent.height,
                             capabilities.maxImageExtent.height);
  return extent;
}

static auto findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                           VkMemoryPropertyFlags properties) -> uint32_t {
  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
    if (((typeFilter & (1 << i)) != 0U) &&
        (memoryProperties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;
};

static auto readFile(const std::string &filename) -> std::vector<char> {
  std::ifstream file{filename, std::ios::ate | std::ios::binary};

  if (!file.is_open())
    throw std::runtime_error("failed to open file!");

  const auto fileSize{file.tellg()};
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

  return buffer;
}

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

  Instance(const Instance &) = delete;
  auto operator=(const Instance &) -> Instance & = delete;
  Instance(Instance &&) = delete;
  auto operator=(Instance &&) -> Instance & = delete;

  VkInstance instance{};
};

struct Device {
  Device(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    const auto indices{std::set<uint32_t>{
        graphicsSupportingQueueFamilyIndex(physicalDevice),
        presentSupportingQueueFamilyIndex(physicalDevice, surface)}};
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos(indices.size());
    const std::array<float, 1> queuePriority{1.0F};
    std::transform(indices.begin(), indices.end(), queueCreateInfos.begin(),
                   [&queuePriority](uint32_t index) {
                     VkDeviceQueueCreateInfo queueCreateInfo{};
                     queueCreateInfo.sType =
                         VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                     queueCreateInfo.queueFamilyIndex = index;
                     queueCreateInfo.queueCount = queuePriority.size();
                     queueCreateInfo.pQueuePriorities = queuePriority.data();
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

  Device(const Device &) = delete;
  auto operator=(const Device &) -> Device & = delete;
  Device(Device &&) = delete;
  auto operator=(Device &&) -> Device & = delete;

  VkDevice device{};
};

struct Surface {
  Surface(VkInstance instance, GLFWwindow *window) : instance{instance} {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create window surface!");
  }

  ~Surface() { vkDestroySurfaceKHR(instance, surface, nullptr); }

  Surface(const Surface &) = delete;
  auto operator=(const Surface &) -> Surface & = delete;
  Surface(Surface &&) = delete;
  auto operator=(Surface &&) -> Surface & = delete;

  VkInstance instance;
  VkSurfaceKHR surface{};
};

struct Swapchain {
  Swapchain(VkDevice device, VkPhysicalDevice physicalDevice,
            VkSurfaceKHR surface, GLFWwindow *window)
      : device{device} {
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;

    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface,
                                              &capabilities);

    auto imageCount{capabilities.minImageCount + 1};
    if (capabilities.maxImageCount > 0 &&
        imageCount > capabilities.maxImageCount) {
      imageCount = capabilities.maxImageCount;
    }

    auto formatCount{vulkanCountFromPhysicalDevice(
        physicalDevice, [&surface](VkPhysicalDevice device_, uint32_t *count) {
          vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, count,
                                               nullptr);
        })};
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                         formats.data());
    const auto surfaceFormat{swapSurfaceFormat(formats)};
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;

    createInfo.minImageCount = imageCount;
    createInfo.imageExtent = swapExtent(physicalDevice, surface, window);
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    const std::array<uint32_t, 2> queueFamilyIndices = {
        graphicsSupportingQueueFamilyIndex(physicalDevice),
        presentSupportingQueueFamilyIndex(physicalDevice, surface)};

    if (queueFamilyIndices[0] != queueFamilyIndices[1]) {
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = queueFamilyIndices.size();
      createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
    } else {
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    auto presentModeCount{vulkanCountFromPhysicalDevice(
        physicalDevice, [&surface](VkPhysicalDevice device_, uint32_t *count) {
          vkGetPhysicalDeviceSurfacePresentModesKHR(device_, surface, count,
                                                    nullptr);
        })};
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        physicalDevice, surface, &presentModeCount, presentModes.data());
    createInfo.presentMode = swapPresentMode(presentModes);

    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.clipped = VK_TRUE;

    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create swap chain!");
  }

  ~Swapchain() { vkDestroySwapchainKHR(device, swapChain, nullptr); }

  Swapchain(const Swapchain &) = delete;
  auto operator=(const Swapchain &) -> Swapchain & = delete;
  Swapchain(Swapchain &&) = delete;
  auto operator=(Swapchain &&) -> Swapchain & = delete;

  VkDevice device;
  VkSwapchainKHR swapChain{};
};

struct ImageView {
  ImageView(VkDevice device, VkPhysicalDevice physicalDevice,
            VkSurfaceKHR surface, VkImage image)
      : device{device} {
    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;

    createInfo.image = image;

    auto formatCount{vulkanCountFromPhysicalDevice(
        physicalDevice, [&surface](VkPhysicalDevice device_, uint32_t *count) {
          vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, count,
                                               nullptr);
        })};
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                         formats.data());
    const auto surfaceFormat{swapSurfaceFormat(formats)};
    createInfo.format = surfaceFormat.format;

    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &createInfo, nullptr, &view) != VK_SUCCESS)
      throw std::runtime_error("failed to create image view!");
  }

  ~ImageView() {
    if (view != nullptr)
      vkDestroyImageView(device, view, nullptr);
  }

  ImageView(ImageView &&other) noexcept
      : device{other.device}, view{other.view} {
    other.view = nullptr;
  }

  auto operator=(ImageView &&) -> ImageView & = delete;
  ImageView(const ImageView &) = delete;
  auto operator=(const ImageView &) -> ImageView & = delete;

  VkDevice device;
  VkImageView view{};
};

struct ShaderModule {
  ShaderModule(VkDevice device, const std::vector<char> &code)
      : device{device} {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    if (vkCreateShaderModule(device, &createInfo, nullptr, &module) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create shader module!");
  }

  ShaderModule(ShaderModule &&) = delete;
  auto operator=(ShaderModule &&) -> ShaderModule & = delete;
  ShaderModule(const ShaderModule &) = delete;
  auto operator=(const ShaderModule &) -> ShaderModule & = delete;

  ~ShaderModule() { vkDestroyShaderModule(device, module, nullptr); }

  VkDevice device;
  VkShaderModule module{};
};

struct PipelineLayout {
  explicit PipelineLayout(VkDevice device) : device{device} {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("failed to create pipeline layout!");
  }

  PipelineLayout(PipelineLayout &&) = delete;
  auto operator=(PipelineLayout &&) -> PipelineLayout & = delete;
  PipelineLayout(const PipelineLayout &) = delete;
  auto operator=(const PipelineLayout &) -> PipelineLayout & = delete;

  ~PipelineLayout() {
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  }

  VkDevice device;
  VkPipelineLayout pipelineLayout{};
};

struct RenderPass {
  RenderPass(VkDevice device, VkPhysicalDevice physicalDevice,
             VkSurfaceKHR surface)
      : device{device} {
    std::array<VkAttachmentDescription, 1> colorAttachment{};

    auto formatCount{vulkanCountFromPhysicalDevice(
        physicalDevice, [&surface](VkPhysicalDevice device_, uint32_t *count) {
          vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, count,
                                               nullptr);
        })};
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                         formats.data());
    colorAttachment.at(0).format = swapSurfaceFormat(formats).format;

    colorAttachment.at(0).samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.at(0).loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.at(0).storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.at(0).stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.at(0).stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.at(0).initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.at(0).finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    std::array<VkAttachmentReference, 1> colorAttachmentRef{};
    colorAttachmentRef.at(0).attachment = 0;
    colorAttachmentRef.at(0).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    std::array<VkSubpassDescription, 1> subpass{};
    subpass.at(0).pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.at(0).colorAttachmentCount = colorAttachmentRef.size();
    subpass.at(0).pColorAttachments = colorAttachmentRef.data();

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = colorAttachment.size();
    renderPassInfo.pAttachments = colorAttachment.data();
    renderPassInfo.subpassCount = subpass.size();
    renderPassInfo.pSubpasses = subpass.data();

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create render pass!");
  }

  ~RenderPass() { vkDestroyRenderPass(device, renderPass, nullptr); }

  RenderPass(RenderPass &&) = delete;
  auto operator=(RenderPass &&) -> RenderPass & = delete;
  RenderPass(const RenderPass &) = delete;
  auto operator=(const RenderPass &) -> RenderPass & = delete;

  VkDevice device;
  VkRenderPass renderPass{};
};

struct Pipeline {
  Pipeline(VkDevice device, VkPhysicalDevice physicalDevice,
           VkSurfaceKHR surface, VkPipelineLayout pipelineLayout,
           VkRenderPass renderPass, const std::string &vertexShaderCodePath,
           const std::string &fragmentShaderCodePath, GLFWwindow *window)
      : device{device} {
    const vulkan_wrappers::ShaderModule vertexShaderModule{
        device, readFile(vertexShaderCodePath)};
    const vulkan_wrappers::ShaderModule fragmentShaderModule{
        device, readFile(fragmentShaderCodePath)};

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertexShaderModule.module;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragmentShaderModule.module;
    fragShaderStageInfo.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
        vertShaderStageInfo, fragShaderStageInfo};
    pipelineInfo.stageCount = shaderStages.size();
    pipelineInfo.pStages = shaderStages.data();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    std::array<VkVertexInputBindingDescription, 1> bindingDescription{};
    bindingDescription.at(0).binding = 0;
    bindingDescription.at(0).stride = sizeof(Vertex);
    bindingDescription.at(0).inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    vertexInputInfo.vertexBindingDescriptionCount = bindingDescription.size();
    vertexInputInfo.vertexAttributeDescriptionCount =
        attributeDescriptions.size();
    vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data();
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;

    std::array<VkViewport, 1> viewport{};
    viewport.at(0).x = 0.0F;
    viewport.at(0).y = 0.0F;

    const auto swapChainExtent{swapExtent(physicalDevice, surface, window)};
    viewport.at(0).width = swapChainExtent.width;
    viewport.at(0).height = swapChainExtent.height;

    viewport.at(0).minDepth = 0.0F;
    viewport.at(0).maxDepth = 1.0F;

    std::array<VkRect2D, 1> scissor{};
    scissor.at(0).offset = {0, 0};
    scissor.at(0).extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = viewport.size();
    viewportState.pViewports = viewport.data();
    viewportState.scissorCount = scissor.size();
    viewportState.pScissors = scissor.data();
    pipelineInfo.pViewportState = &viewportState;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0F;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    pipelineInfo.pRasterizationState = &rasterizer;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    pipelineInfo.pMultisampleState = &multisampling;

    std::array<VkPipelineColorBlendAttachmentState, 1> colorBlendAttachment{};
    colorBlendAttachment.at(0).colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.at(0).blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = colorBlendAttachment.size();
    colorBlending.pAttachments = colorBlendAttachment.data();
    colorBlending.blendConstants[0] = 0.0F;
    colorBlending.blendConstants[1] = 0.0F;
    colorBlending.blendConstants[2] = 0.0F;
    colorBlending.blendConstants[3] = 0.0F;
    pipelineInfo.pColorBlendState = &colorBlending;

    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                  nullptr, &pipeline) != VK_SUCCESS)
      throw std::runtime_error("failed to create graphics pipeline!");
  }

  ~Pipeline() { vkDestroyPipeline(device, pipeline, nullptr); }

  Pipeline(Pipeline &&) = delete;
  auto operator=(Pipeline &&) -> Pipeline & = delete;
  Pipeline(const Pipeline &) = delete;
  auto operator=(const Pipeline &) -> Pipeline & = delete;

  VkDevice device;
  VkPipeline pipeline{};
};

struct Framebuffer {
  Framebuffer(VkDevice device, VkPhysicalDevice physicalDevice,
              VkSurfaceKHR surface, VkRenderPass renderPass,
              VkImageView imageView, GLFWwindow *window)
      : device{device} {
    std::array<VkImageView, 1> attachments = {imageView};

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = attachments.size();
    framebufferInfo.pAttachments = attachments.data();

    const auto swapChainExtent{swapExtent(physicalDevice, surface, window)};
    framebufferInfo.width = swapChainExtent.width;
    framebufferInfo.height = swapChainExtent.height;

    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffer) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create framebuffer!");
    }
  }

  ~Framebuffer() {
    if (framebuffer != nullptr)
      vkDestroyFramebuffer(device, framebuffer, nullptr);
  }

  Framebuffer(Framebuffer &&other) noexcept
      : device{other.device}, framebuffer{other.framebuffer} {
    other.framebuffer = nullptr;
  }

  auto operator=(Framebuffer &&) -> Framebuffer & = delete;
  Framebuffer(const Framebuffer &) = delete;
  auto operator=(const Framebuffer &) -> Framebuffer & = delete;

  VkDevice device;
  VkFramebuffer framebuffer{};
};

struct CommandPool {
  CommandPool(VkDevice device, VkPhysicalDevice physicalDevice)
      : device{device} {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex =
        graphicsSupportingQueueFamilyIndex(physicalDevice);
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create command pool!");
  }

  ~CommandPool() { vkDestroyCommandPool(device, commandPool, nullptr); }

  CommandPool(CommandPool &&) = delete;
  auto operator=(CommandPool &&) -> CommandPool & = delete;
  CommandPool(const CommandPool &) = delete;
  auto operator=(const CommandPool &) -> CommandPool & = delete;

  VkDevice device;
  VkCommandPool commandPool{};
};

struct Semaphore {
  explicit Semaphore(VkDevice device) : device{device} {
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create semaphore!");
  }

  ~Semaphore() {
    if (semaphore != nullptr)
      vkDestroySemaphore(device, semaphore, nullptr);
  }

  Semaphore(Semaphore &&other) noexcept
      : device{other.device}, semaphore{other.semaphore} {
    other.semaphore = nullptr;
  }

  auto operator=(Semaphore &&) -> Semaphore & = delete;
  Semaphore(const Semaphore &) = delete;
  auto operator=(const Semaphore &) -> Semaphore & = delete;

  VkDevice device;
  VkSemaphore semaphore{};
};

struct Fence {
  explicit Fence(VkDevice device) : device{device} {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
      throw std::runtime_error("failed to create fence!");
  }

  ~Fence() {
    if (fence != nullptr)
      vkDestroyFence(device, fence, nullptr);
  }

  Fence(Fence &&other) noexcept : device{other.device}, fence{other.fence} {
    other.fence = nullptr;
  }

  auto operator=(Fence &&) -> Fence & = delete;
  Fence(const Fence &) = delete;
  auto operator=(const Fence &) -> Fence & = delete;

  VkDevice device;
  VkFence fence{};
};

struct CommandBuffers {
  CommandBuffers(VkDevice device, VkCommandPool commandPool,
                 std::vector<VkCommandBuffer>::size_type size)
      : device{device}, commandPool{commandPool}, commandBuffers(size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;

    allocInfo.commandPool = commandPool;

    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to allocate command buffers!");
  }

  ~CommandBuffers() {
    vkFreeCommandBuffers(device, commandPool,
                         static_cast<uint32_t>(commandBuffers.size()),
                         commandBuffers.data());
  }

  CommandBuffers(CommandBuffers &&) = delete;
  auto operator=(CommandBuffers &&) -> CommandBuffers & = delete;
  CommandBuffers(const CommandBuffers &) = delete;
  auto operator=(const CommandBuffers &) -> CommandBuffers & = delete;

  VkDevice device;
  VkCommandPool commandPool;
  std::vector<VkCommandBuffer> commandBuffers;
};

struct Buffer {
  Buffer(VkDevice device, VkBufferUsageFlags usage, VkDeviceSize bufferSize)
      : device{device} {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
      throw std::runtime_error("failed to create buffer!");
  }

  ~Buffer() { vkDestroyBuffer(device, buffer, nullptr); }

  Buffer(Buffer &&) = delete;
  auto operator=(Buffer &&) -> Buffer & = delete;
  Buffer(const Buffer &) = delete;
  auto operator=(const Buffer &) -> Buffer & = delete;

  VkDevice device;
  VkBuffer buffer{};
};

struct DeviceMemory {
  DeviceMemory(VkDevice device, VkPhysicalDevice physicalDevice,
               VkBuffer buffer, VkMemoryPropertyFlags properties)
      : device{device} {
    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memoryRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(
        physicalDevice, memoryRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
      throw std::runtime_error("failed to allocate vertex buffer memory!");
  }

  ~DeviceMemory() { vkFreeMemory(device, memory, nullptr); }

  DeviceMemory(DeviceMemory &&) = delete;
  auto operator=(DeviceMemory &&) -> DeviceMemory & = delete;
  DeviceMemory(const DeviceMemory &) = delete;
  auto operator=(const DeviceMemory &) -> DeviceMemory & = delete;

  VkDevice device;
  VkDeviceMemory memory{};
};
} // namespace vulkan_wrappers

constexpr auto windowWidth{800};
constexpr auto windowHeight{600};

static auto suitable(VkPhysicalDevice device, VkSurfaceKHR surface) -> bool {
  auto extensionPropertyCount{vulkanCountFromPhysicalDevice(
      device, [](VkPhysicalDevice device_, uint32_t *count) {
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

  if (vulkanCountFromPhysicalDevice(
          device, [&surface](VkPhysicalDevice device_, uint32_t *count) {
            vkGetPhysicalDeviceSurfacePresentModesKHR(device_, surface, count,
                                                      nullptr);
          }) == 0)
    return false;

  if (vulkanCountFromPhysicalDevice(device, [&surface](VkPhysicalDevice device_,
                                                       uint32_t *count) {
        vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, count, nullptr);
      }) == 0)
    return false;

  uint32_t index{0U};
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

static auto vulkanDevices(VkInstance instance)
    -> std::vector<VkPhysicalDevice> {
  uint32_t deviceCount{0};
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
  return devices;
}

static auto swapChainImages(VkDevice device, VkSwapchainKHR swapChain)
    -> std::vector<VkImage> {
  uint32_t imageCount{0};
  vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
  std::vector<VkImage> swapChainImages(imageCount);
  vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
                          swapChainImages.data());
  return swapChainImages;
}

static void framebufferResizeCallback(GLFWwindow *window, int width,
                                      int height) {
  auto *framebufferResized =
      static_cast<bool *>(glfwGetWindowUserPointer(window));
  *framebufferResized = true;
}

static void prepareForSwapChainRecreation(VkDevice device, GLFWwindow *window) {
  int width{0};
  int height{0};
  glfwGetFramebufferSize(window, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
  }

  vkDeviceWaitIdle(device);
}

static void copyBuffer(VkDevice device, VkCommandPool commandPool,
                       VkQueue graphicsQueue, VkBuffer srcBuffer,
                       VkBuffer dstBuffer, VkDeviceSize size) {
  vulkan_wrappers::CommandBuffers vulkanCommandBuffers{device, commandPool, 1};

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(vulkanCommandBuffers.commandBuffers.at(0), &beginInfo);

  VkBufferCopy copyRegion{};
  copyRegion.size = size;
  vkCmdCopyBuffer(vulkanCommandBuffers.commandBuffers.at(0), srcBuffer,
                  dstBuffer, 1, &copyRegion);

  vkEndCommandBuffer(vulkanCommandBuffers.commandBuffers.at(0));

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount =
      static_cast<uint32_t>(vulkanCommandBuffers.commandBuffers.size());
  submitInfo.pCommandBuffers = vulkanCommandBuffers.commandBuffers.data();

  vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphicsQueue);
}

static void run(const std::string &vertexShaderCodePath,
                const std::string &fragmentShaderCodePath) {
  const glfw_wrappers::Init glfwInitialization;

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  const glfw_wrappers::Window glfwWindow{windowWidth, windowHeight};
  auto framebufferResized{false};
  glfwSetWindowUserPointer(glfwWindow.window, &framebufferResized);
  glfwSetFramebufferSizeCallback(glfwWindow.window, framebufferResizeCallback);

  const vulkan_wrappers::Instance vulkanInstance;
  const vulkan_wrappers::Surface vulkanSurface{vulkanInstance.instance,
                                               glfwWindow.window};
  auto *vulkanPhysicalDevice{suitableDevice(
      vulkanDevices(vulkanInstance.instance), vulkanSurface.surface)};
  const vulkan_wrappers::Device vulkanDevice{vulkanPhysicalDevice,
                                             vulkanSurface.surface};

  VkQueue graphicsQueue{nullptr};
  vkGetDeviceQueue(vulkanDevice.device,
                   graphicsSupportingQueueFamilyIndex(vulkanPhysicalDevice), 0,
                   &graphicsQueue);
  VkQueue presentQueue{nullptr};
  vkGetDeviceQueue(vulkanDevice.device,
                   presentSupportingQueueFamilyIndex(vulkanPhysicalDevice,
                                                     vulkanSurface.surface),
                   0, &presentQueue);

  const vulkan_wrappers::CommandPool vulkanCommandPool{vulkanDevice.device,
                                                       vulkanPhysicalDevice};

  const std::vector<Vertex> vertices = {{{-0.5F, -0.5F}, {1.0F, 0.0F, 0.0F}},
                                        {{0.5F, -0.5F}, {0.0F, 1.0F, 0.0F}},
                                        {{0.5F, 0.5F}, {0.0F, 0.0F, 1.0F}},
                                        {{-0.5F, 0.5F}, {1.0F, 1.0F, 1.0F}}};

  const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

  VkDeviceSize vertexBufferSize{sizeof(vertices[0]) * vertices.size()};

  const vulkan_wrappers::Buffer vulkanVertexBuffer{
      vulkanDevice.device,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      vertexBufferSize};
  const vulkan_wrappers::DeviceMemory vulkanVertexBufferMemory{
      vulkanDevice.device, vulkanPhysicalDevice, vulkanVertexBuffer.buffer,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
  vkBindBufferMemory(vulkanDevice.device, vulkanVertexBuffer.buffer,
                     vulkanVertexBufferMemory.memory, 0);
  {
    const vulkan_wrappers::Buffer vulkanStagingBuffer{
        vulkanDevice.device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        vertexBufferSize};
    const vulkan_wrappers::DeviceMemory vulkanStagingBufferMemory{
        vulkanDevice.device, vulkanPhysicalDevice, vulkanStagingBuffer.buffer,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    vkBindBufferMemory(vulkanDevice.device, vulkanStagingBuffer.buffer,
                       vulkanStagingBufferMemory.memory, 0);

    void *data = nullptr;
    vkMapMemory(vulkanDevice.device, vulkanStagingBufferMemory.memory, 0,
                vertexBufferSize, 0, &data);
    memcpy(data, vertices.data(), vertexBufferSize);
    vkUnmapMemory(vulkanDevice.device, vulkanStagingBufferMemory.memory);

    copyBuffer(vulkanDevice.device, vulkanCommandPool.commandPool,
               graphicsQueue, vulkanStagingBuffer.buffer,
               vulkanVertexBuffer.buffer, vertexBufferSize);
  }

  VkDeviceSize indexBufferSize{sizeof(indices[0]) * indices.size()};

  const vulkan_wrappers::Buffer vulkanIndexBuffer{
      vulkanDevice.device,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      indexBufferSize};
  const vulkan_wrappers::DeviceMemory vulkanIndexBufferMemory{
      vulkanDevice.device, vulkanPhysicalDevice, vulkanIndexBuffer.buffer,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
  vkBindBufferMemory(vulkanDevice.device, vulkanIndexBuffer.buffer,
                     vulkanIndexBufferMemory.memory, 0);
  {
    const vulkan_wrappers::Buffer vulkanStagingBuffer{
        vulkanDevice.device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, indexBufferSize};
    const vulkan_wrappers::DeviceMemory vulkanStagingBufferMemory{
        vulkanDevice.device, vulkanPhysicalDevice, vulkanStagingBuffer.buffer,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    vkBindBufferMemory(vulkanDevice.device, vulkanStagingBuffer.buffer,
                       vulkanStagingBufferMemory.memory, 0);

    void *data = nullptr;
    vkMapMemory(vulkanDevice.device, vulkanStagingBufferMemory.memory, 0,
                indexBufferSize, 0, &data);
    memcpy(data, indices.data(), indexBufferSize);
    vkUnmapMemory(vulkanDevice.device, vulkanStagingBufferMemory.memory);

    copyBuffer(vulkanDevice.device, vulkanCommandPool.commandPool,
               graphicsQueue, vulkanStagingBuffer.buffer,
               vulkanIndexBuffer.buffer, indexBufferSize);
  }

  const auto maxFramesInFlight{2};

  std::vector<vulkan_wrappers::Semaphore> vulkanImageAvailableSemaphores;
  std::vector<vulkan_wrappers::Semaphore> vulkanRenderFinishedSemaphores;
  std::vector<vulkan_wrappers::Fence> vulkanInFlightFences;
  for (auto i{0}; i < maxFramesInFlight; ++i) {
    vulkanImageAvailableSemaphores.emplace_back(vulkanDevice.device);
    vulkanRenderFinishedSemaphores.emplace_back(vulkanDevice.device);
    vulkanInFlightFences.emplace_back(vulkanDevice.device);
  }

  std::vector<VkFence> imagesInFlight;

  auto currentFrame{0};
  auto playing{true};
  while (playing) {
    const vulkan_wrappers::Swapchain vulkanSwapchain{
        vulkanDevice.device, vulkanPhysicalDevice, vulkanSurface.surface,
        glfwWindow.window};

    const auto swapChainImages{
        ::swapChainImages(vulkanDevice.device, vulkanSwapchain.swapChain)};
    std::vector<vulkan_wrappers::ImageView> swapChainImageViews;
    std::transform(
        swapChainImages.begin(), swapChainImages.end(),
        std::back_inserter(swapChainImageViews),
        [&vulkanDevice, vulkanPhysicalDevice, &vulkanSurface](VkImage image) {
          return vulkan_wrappers::ImageView{vulkanDevice.device,
                                            vulkanPhysicalDevice,
                                            vulkanSurface.surface, image};
        });

    const vulkan_wrappers::RenderPass vulkanRenderPass{
        vulkanDevice.device, vulkanPhysicalDevice, vulkanSurface.surface};

    const vulkan_wrappers::PipelineLayout vulkanPipelineLayout{
        vulkanDevice.device};

    const vulkan_wrappers::Pipeline vulkanPipeline{
        vulkanDevice.device,         vulkanPhysicalDevice,
        vulkanSurface.surface,       vulkanPipelineLayout.pipelineLayout,
        vulkanRenderPass.renderPass, vertexShaderCodePath,
        fragmentShaderCodePath,      glfwWindow.window};

    std::vector<vulkan_wrappers::Framebuffer> vulkanFrameBuffers;
    std::transform(swapChainImageViews.begin(), swapChainImageViews.end(),
                   std::back_inserter(vulkanFrameBuffers),
                   [&vulkanDevice, vulkanPhysicalDevice, &vulkanSurface,
                    &vulkanRenderPass,
                    &glfwWindow](const vulkan_wrappers::ImageView &imageView) {
                     return vulkan_wrappers::Framebuffer{
                         vulkanDevice.device,   vulkanPhysicalDevice,
                         vulkanSurface.surface, vulkanRenderPass.renderPass,
                         imageView.view,        glfwWindow.window};
                   });
    const vulkan_wrappers::CommandBuffers vulkanCommandBuffers{
        vulkanDevice.device, vulkanCommandPool.commandPool,
        vulkanFrameBuffers.size()};

    for (auto i{0}; i < vulkanCommandBuffers.commandBuffers.size(); i++) {
      VkCommandBufferBeginInfo beginInfo{};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      if (vkBeginCommandBuffer(vulkanCommandBuffers.commandBuffers[i],
                               &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording command buffer!");

      VkRenderPassBeginInfo renderPassInfo{};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      renderPassInfo.renderPass = vulkanRenderPass.renderPass;
      renderPassInfo.framebuffer = vulkanFrameBuffers.at(i).framebuffer;
      renderPassInfo.renderArea.offset = {0, 0};

      renderPassInfo.renderArea.extent = swapExtent(
          vulkanPhysicalDevice, vulkanSurface.surface, glfwWindow.window);

      const std::array<VkClearValue, 1> clearColor = {
          {{{{0.0F, 0.0F, 0.0F, 1.0F}}}}};
      renderPassInfo.clearValueCount = clearColor.size();
      renderPassInfo.pClearValues = clearColor.data();

      vkCmdBeginRenderPass(vulkanCommandBuffers.commandBuffers[i],
                           &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

      vkCmdBindPipeline(vulkanCommandBuffers.commandBuffers[i],
                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                        vulkanPipeline.pipeline);

      std::array<VkBuffer, 1> vertexBuffers = {vulkanVertexBuffer.buffer};
      std::array<VkDeviceSize, 1> offsets = {0};
      vkCmdBindVertexBuffers(vulkanCommandBuffers.commandBuffers[i], 0, 1,
                             vertexBuffers.data(), offsets.data());

      vkCmdBindIndexBuffer(vulkanCommandBuffers.commandBuffers[i],
                           vulkanIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

      vkCmdDrawIndexed(vulkanCommandBuffers.commandBuffers[i],
                       static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

      vkCmdEndRenderPass(vulkanCommandBuffers.commandBuffers[i]);

      if (vkEndCommandBuffer(vulkanCommandBuffers.commandBuffers[i]) !=
          VK_SUCCESS)
        throw std::runtime_error("failed to record command buffer!");
    }

    imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
    auto recreatingSwapChain{false};
    while (!recreatingSwapChain) {
      if (glfwWindowShouldClose(glfwWindow.window) != 0) {
        playing = false;
        break;
      }

      glfwPollEvents();
      vkWaitForFences(vulkanDevice.device, 1,
                      &vulkanInFlightFences[currentFrame].fence, VK_TRUE,
                      UINT64_MAX);

      uint32_t imageIndex{0};
      {
        const auto result{vkAcquireNextImageKHR(
            vulkanDevice.device, vulkanSwapchain.swapChain, UINT64_MAX,
            vulkanImageAvailableSemaphores[currentFrame].semaphore,
            VK_NULL_HANDLE, &imageIndex)};

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
          prepareForSwapChainRecreation(vulkanDevice.device, glfwWindow.window);
          recreatingSwapChain = true;
          continue;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
          throw std::runtime_error("failed to acquire swap chain image!");
      }

      if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        vkWaitForFences(vulkanDevice.device, 1, &imagesInFlight[imageIndex],
                        VK_TRUE, UINT64_MAX);

      imagesInFlight[imageIndex] = vulkanInFlightFences[currentFrame].fence;

      VkSubmitInfo submitInfo{};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

      const std::array<VkSemaphore, 1> waitSemaphores = {
          vulkanImageAvailableSemaphores[currentFrame].semaphore};
      const std::array<VkPipelineStageFlags, 1> waitStages = {
          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
      submitInfo.waitSemaphoreCount = waitSemaphores.size();
      submitInfo.pWaitSemaphores = waitSemaphores.data();
      submitInfo.pWaitDstStageMask = waitStages.data();

      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers =
          &vulkanCommandBuffers.commandBuffers[imageIndex];

      const std::array<VkSemaphore, 1> signalSemaphores = {
          vulkanRenderFinishedSemaphores[currentFrame].semaphore};
      submitInfo.signalSemaphoreCount = signalSemaphores.size();
      submitInfo.pSignalSemaphores = signalSemaphores.data();

      vkResetFences(vulkanDevice.device, 1,
                    &vulkanInFlightFences[currentFrame].fence);

      if (vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                        vulkanInFlightFences[currentFrame].fence) != VK_SUCCESS)
        throw std::runtime_error("failed to submit draw command buffer!");

      VkPresentInfoKHR presentInfo{};
      presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

      presentInfo.waitSemaphoreCount = waitSemaphores.size();
      presentInfo.pWaitSemaphores = signalSemaphores.data();

      const std::array<VkSwapchainKHR, 1> swapChains = {
          vulkanSwapchain.swapChain};
      presentInfo.swapchainCount = swapChains.size();
      presentInfo.pSwapchains = swapChains.data();

      presentInfo.pImageIndices = &imageIndex;

      {
        const auto result{vkQueuePresentKHR(presentQueue, &presentInfo)};

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
            framebufferResized) {
          framebufferResized = false;
          prepareForSwapChainRecreation(vulkanDevice.device, glfwWindow.window);
          recreatingSwapChain = true;
        } else if (result != VK_SUCCESS)
          throw std::runtime_error("failed to present swap chain image!");
      }

      if (++currentFrame == maxFramesInFlight)
        currentFrame = 0;
    }
  }

  vkDeviceWaitIdle(vulkanDevice.device);
}

int main(int argc, char *argv[]) {
  const std::span<char *> arguments{
      argv, static_cast<std::span<char *>::size_type>(argc)};
  if (arguments.size() < 3)
    return EXIT_FAILURE;
  try {
    run(arguments[1], arguments[2]);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
