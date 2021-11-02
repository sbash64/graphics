#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
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

    auto imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 &&
        imageCount > capabilities.maxImageCount) {
      imageCount = capabilities.maxImageCount;
    }

    auto formatCount{vulkanCount(
        physicalDevice, [&surface](VkPhysicalDevice device_, uint32_t *count) {
          vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, count,
                                               nullptr);
        })};
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                         formats.data());

    const auto surfaceFormat{swapSurfaceFormat(formats)};

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = swapExtent(capabilities, window);
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    const std::array<uint32_t, 2> queueFamilyIndices = {
        graphicsSupportingQueueFamilyIndex(physicalDevice),
        presentSupportingQueueFamilyIndex(physicalDevice, surface)};

    if (queueFamilyIndices[0] != queueFamilyIndices[1]) {
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
    } else {
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    auto presentModeCount{vulkanCount(
        physicalDevice, [&surface](VkPhysicalDevice device_, uint32_t *count) {
          vkGetPhysicalDeviceSurfacePresentModesKHR(device_, surface, count,
                                                    nullptr);
        })};
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        physicalDevice, surface, &presentModeCount, presentModes.data());

    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = swapPresentMode(presentModes);
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
    createInfo.image = image;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;

    auto formatCount{vulkanCount(
        physicalDevice, [&surface](VkPhysicalDevice device_, uint32_t *count) {
          vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, count,
                                               nullptr);
        })};
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                         formats.data());
    auto surfaceFormat = swapSurfaceFormat(formats);
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
        VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module!");
    }
  }

  ~ShaderModule() { vkDestroyShaderModule(device, module, nullptr); }

  VkDevice device;
  VkShaderModule module{};
};

struct PipelineLayout {
  PipelineLayout(VkDevice device) : device{device} {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("failed to create pipeline layout!");
  }

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
    VkAttachmentDescription colorAttachment{};

    auto formatCount{vulkanCount(
        physicalDevice, [&surface](VkPhysicalDevice device_, uint32_t *count) {
          vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, count,
                                               nullptr);
        })};
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                         formats.data());
    colorAttachment.format = swapSurfaceFormat(formats).format;

    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create render pass!");
  }

  ~RenderPass() { vkDestroyRenderPass(device, renderPass, nullptr); }

  VkDevice device;
  VkRenderPass renderPass{};
};

struct Pipeline {
  Pipeline(VkDevice device, VkPhysicalDevice physicalDevice,
           VkSurfaceKHR surface, VkPipelineLayout pipelineLayout,
           VkRenderPass renderPass, VkShaderModule vertexShaderModule,
           VkShaderModule fragmentShaderModule, GLFWwindow *window)
      : device{device} {
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertexShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragmentShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};
    pipelineInfo.pStages = shaderStages;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;

    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface,
                                              &capabilities);
    const auto swapChainExtent{swapExtent(capabilities, window)};

    VkViewport viewport{};
    viewport.x = 0.0F;
    viewport.y = 0.0F;
    viewport.width = swapChainExtent.width;
    viewport.height = swapChainExtent.height;
    viewport.minDepth = 0.0F;
    viewport.maxDepth = 1.0F;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;
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

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
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

    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface,
                                              &capabilities);
    const auto swapChainExtent{swapExtent(capabilities, window)};
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

  if (vulkanCount(device,
                  [&surface](VkPhysicalDevice device_, uint32_t *count) {
                    vkGetPhysicalDeviceSurfacePresentModesKHR(device_, surface,
                                                              count, nullptr);
                  }) == 0)
    return false;

  if (vulkanCount(device, [&surface](VkPhysicalDevice device_,
                                     uint32_t *count) {
        vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, count, nullptr);
      }) == 0)
    return false;

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

static auto readFile(const std::string &filename) -> std::vector<char> {
  std::ifstream file{filename, std::ios::ate | std::ios::binary};

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  const auto fileSize = file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

  return buffer;
}

static void run(const std::string &vertexShaderCodePath,
                const std::string &fragmentShaderCodePath) {
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

  VkQueue graphicsQueue{nullptr};
  vkGetDeviceQueue(vulkanDevice.device,
                   graphicsSupportingQueueFamilyIndex(vulkanPhysicalDevice), 0,
                   &graphicsQueue);
  VkQueue presentQueue{nullptr};
  vkGetDeviceQueue(vulkanDevice.device,
                   presentSupportingQueueFamilyIndex(vulkanPhysicalDevice,
                                                     vulkanSurface.surface),
                   0, &presentQueue);

  vulkan_wrappers::Swapchain vulkanSwapchain{
      vulkanDevice.device, vulkanPhysicalDevice, vulkanSurface.surface,
      glfwWindow.window};

  uint32_t imageCount{0};
  vkGetSwapchainImagesKHR(vulkanDevice.device, vulkanSwapchain.swapChain,
                          &imageCount, nullptr);
  std::vector<VkImage> swapChainImages(imageCount);
  vkGetSwapchainImagesKHR(vulkanDevice.device, vulkanSwapchain.swapChain,
                          &imageCount, swapChainImages.data());

  std::vector<vulkan_wrappers::ImageView> swapChainImageViews;
  std::transform(
      swapChainImages.begin(), swapChainImages.end(),
      std::back_inserter(swapChainImageViews),
      [&vulkanDevice, &vulkanPhysicalDevice, &vulkanSurface](VkImage image) {
        return vulkan_wrappers::ImageView{vulkanDevice.device,
                                          vulkanPhysicalDevice,
                                          vulkanSurface.surface, image};
      });

  vulkan_wrappers::RenderPass vulkanRenderPass{
      vulkanDevice.device, vulkanPhysicalDevice, vulkanSurface.surface};

  vulkan_wrappers::ShaderModule vertexShaderModule{
      vulkanDevice.device, readFile(vertexShaderCodePath)};
  vulkan_wrappers::ShaderModule fragmentShaderModule{
      vulkanDevice.device, readFile(fragmentShaderCodePath)};

  vulkan_wrappers::PipelineLayout vulkanPipelineLayout{vulkanDevice.device};

  vulkan_wrappers::Pipeline vulkanPipeline{
      vulkanDevice.device,         vulkanPhysicalDevice,
      vulkanSurface.surface,       vulkanPipelineLayout.pipelineLayout,
      vulkanRenderPass.renderPass, vertexShaderModule.module,
      fragmentShaderModule.module, glfwWindow.window};

  std::vector<vulkan_wrappers::Framebuffer> vulkanFrameBuffers;
  std::transform(swapChainImageViews.begin(), swapChainImageViews.end(),
                 std::back_inserter(vulkanFrameBuffers),
                 [&vulkanDevice, &vulkanPhysicalDevice, &vulkanSurface,
                  &vulkanRenderPass,
                  &glfwWindow](const vulkan_wrappers::ImageView &imageView) {
                   return vulkan_wrappers::Framebuffer{
                       vulkanDevice.device,   vulkanPhysicalDevice,
                       vulkanSurface.surface, vulkanRenderPass.renderPass,
                       imageView.view,        glfwWindow.window};
                 });

  vulkan_wrappers::CommandPool vulkanCommandPool{vulkanDevice.device,
                                                 vulkanPhysicalDevice};
  std::vector<VkCommandBuffer> commandBuffers(vulkanFrameBuffers.size());

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = vulkanCommandPool.commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = commandBuffers.size();

  if (vkAllocateCommandBuffers(vulkanDevice.device, &allocInfo,
                               commandBuffers.data()) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }

  for (size_t i = 0; i < commandBuffers.size(); i++) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
      throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = vulkanRenderPass.renderPass;
    renderPassInfo.framebuffer = vulkanFrameBuffers.at(i).framebuffer;
    renderPassInfo.renderArea.offset = {0, 0};

    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        vulkanPhysicalDevice, vulkanSurface.surface, &capabilities);
    const auto swapChainExtent{swapExtent(capabilities, glfwWindow.window)};
    renderPassInfo.renderArea.extent = swapChainExtent;

    VkClearValue clearColor = {{{0.0F, 0.0F, 0.0F, 1.0F}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo,
                         VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                      vulkanPipeline.pipeline);

    vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers[i]);

    if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to record command buffer!");
    }
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

  std::vector<VkFence> imagesInFlight(swapChainImages.size(), VK_NULL_HANDLE);

  size_t currentFrame = 0;
  while (glfwWindowShouldClose(glfwWindow.window) == 0) {
    glfwPollEvents();
    vkWaitForFences(vulkanDevice.device, 1,
                    &vulkanInFlightFences[currentFrame].fence, VK_TRUE,
                    UINT64_MAX);

    uint32_t imageIndex = 0;
    vkAcquireNextImageKHR(
        vulkanDevice.device, vulkanSwapchain.swapChain, UINT64_MAX,
        vulkanImageAvailableSemaphores[currentFrame].semaphore, VK_NULL_HANDLE,
        &imageIndex);

    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
      vkWaitForFences(vulkanDevice.device, 1, &imagesInFlight[imageIndex],
                      VK_TRUE, UINT64_MAX);
    }
    imagesInFlight[imageIndex] = vulkanInFlightFences[currentFrame].fence;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    std::array<VkSemaphore, 1> waitSemaphores = {
        vulkanImageAvailableSemaphores[currentFrame].semaphore};
    std::array<VkPipelineStageFlags, 1> waitStages = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = waitSemaphores.size();
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages.data();

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

    std::array<VkSemaphore, 1> signalSemaphores = {
        vulkanRenderFinishedSemaphores[currentFrame].semaphore};
    submitInfo.signalSemaphoreCount = signalSemaphores.size();
    submitInfo.pSignalSemaphores = signalSemaphores.data();

    vkResetFences(vulkanDevice.device, 1,
                  &vulkanInFlightFences[currentFrame].fence);

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                      vulkanInFlightFences[currentFrame].fence) != VK_SUCCESS) {
      throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = waitSemaphores.size();
    presentInfo.pWaitSemaphores = signalSemaphores.data();

    std::array<VkSwapchainKHR, 1> swapChains = {vulkanSwapchain.swapChain};
    presentInfo.swapchainCount = swapChains.size();
    presentInfo.pSwapchains = swapChains.data();

    presentInfo.pImageIndices = &imageIndex;

    vkQueuePresentKHR(presentQueue, &presentInfo);

    if (++currentFrame == maxFramesInFlight)
      currentFrame = 0;
  }

  vkDeviceWaitIdle(vulkanDevice.device);
}

int main(int argc, char *argv[]) {
  std::span<char *> arguments{argv,
                              static_cast<std::span<char *>::size_type>(argc)};
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
