#include <sbash64/graphics/load-object.hpp>
#include <sbash64/graphics/vulkan-wrappers.hpp>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <set>
#include <stdexcept>

namespace sbash64::graphics {
static auto vulkanCountFromPhysicalDevice(
    VkPhysicalDevice device,
    const std::function<void(VkPhysicalDevice, uint32_t *)> &f) -> uint32_t {
  uint32_t count{0};
  f(device, &count);
  return count;
}

auto swapSurfaceFormat(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
    -> VkSurfaceFormatKHR {
  // auto surface_priority_list = std::vector<VkSurfaceFormatKHR>{
  //     {VK_FORMAT_R8G8B8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
  //     {VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
  //     {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
  //     {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}};
  auto formatCount{vulkanCountFromPhysicalDevice(
      physicalDevice, [&surface](VkPhysicalDevice device_, uint32_t *count) {
        vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, count, nullptr);
      })};
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                       formats.data());
  for (const auto &format : formats)
    if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
      return format;
  return formats.front();
}

static auto swapPresentMode(const std::vector<VkPresentModeKHR> &presentModes)
    -> VkPresentModeKHR {
  // std::vector<VkPresentModeKHR> presentModePriority {
  //   VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_FIFO_KHR,
  //       VK_PRESENT_MODE_IMMEDIATE_KHR,
  // };
  for (const auto &mode : presentModes)
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
      return mode;
  return VK_PRESENT_MODE_FIFO_KHR;
}

static auto findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                           VkMemoryPropertyFlags properties) -> uint32_t {
  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
    if (((typeFilter & (1U << i)) != 0U) &&
        (memoryProperties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

static auto maxUsableSampleCount(VkPhysicalDevice physicalDevice)
    -> VkSampleCountFlagBits {
  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(physicalDevice, &properties);
  const auto counts{properties.limits.framebufferColorSampleCounts &
                    properties.limits.framebufferDepthSampleCounts};
  for (const auto count :
       {VK_SAMPLE_COUNT_64_BIT, VK_SAMPLE_COUNT_32_BIT, VK_SAMPLE_COUNT_16_BIT,
        VK_SAMPLE_COUNT_8_BIT, VK_SAMPLE_COUNT_4_BIT, VK_SAMPLE_COUNT_2_BIT})
    if ((counts & count) != 0U)
      return count;
  return VK_SAMPLE_COUNT_1_BIT;
}

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
Instance::Instance() {
  VkApplicationInfo applicationInfo{};
  applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  applicationInfo.pApplicationName = "Nope";
  applicationInfo.applicationVersion = 0;
  applicationInfo.pEngineName = "N/A";
  applicationInfo.engineVersion = 0;
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

  throwOnError(
      [&]() { return vkCreateInstance(&createInfo, nullptr, &instance); },
      "failed to create instance!");
}

Instance::~Instance() { vkDestroyInstance(instance, nullptr); }

Device::Device(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
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

  VkPhysicalDeviceFeatures supportedFeatures;
  vkGetPhysicalDeviceFeatures(physicalDevice, &supportedFeatures);

  VkPhysicalDeviceFeatures deviceFeatures{};
  if (supportedFeatures.samplerAnisotropy == VK_TRUE)
    deviceFeatures.samplerAnisotropy = VK_TRUE;
  if (supportedFeatures.textureCompressionASTC_LDR == VK_TRUE)
    deviceFeatures.textureCompressionASTC_LDR = VK_TRUE;

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

  throwOnError(
      [&]() {
        return vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);
      },
      "failed to create logical device!");
}

Device::~Device() { vkDestroyDevice(device, nullptr); }

Surface::Surface(VkInstance instance, GLFWwindow *window) : instance{instance} {
  throwOnError(
      [&]() {
        return glfwCreateWindowSurface(instance, window, nullptr, &surface);
      },
      "failed to create window surface!");
}

Surface::~Surface() { vkDestroySurfaceKHR(instance, surface, nullptr); }

Swapchain::Swapchain(VkDevice device, VkPhysicalDevice physicalDevice,
                     VkSurfaceKHR surface, GLFWwindow *window)
    : device{device} {
  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = surface;

  VkSurfaceCapabilitiesKHR capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface,
                                            &capabilities);

  auto imageCount{3U};
  if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
    imageCount = capabilities.maxImageCount;
  if (imageCount < capabilities.minImageCount)
    imageCount = capabilities.minImageCount;

  const auto surfaceFormat{swapSurfaceFormat(physicalDevice, surface)};
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;

  createInfo.minImageCount = imageCount;
  createInfo.imageExtent = swapExtent(physicalDevice, surface, window);
  createInfo.imageArrayLayers = 1;

  // const std::set<VkImageUsageFlagBits> &image_usage_flags = {
  //     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
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

  // const VkSurfaceTransformFlagBitsKHR transform =
  //     VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  createInfo.preTransform = capabilities.currentTransform;

  // VkCompositeAlphaFlagBitsKHR request_composite_alpha{
  //     VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR};
  // static const std::vector<VkCompositeAlphaFlagBitsKHR> composite_alpha_flags
  // =
  //     {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
  //      VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
  //      VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
  //      VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR};
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.clipped = VK_TRUE;

  createInfo.oldSwapchain = VK_NULL_HANDLE;

  throwOnError(
      [&]() {
        return vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain);
      },
      "failed to create swap chain!");
}

Swapchain::~Swapchain() { vkDestroySwapchainKHR(device, swapChain, nullptr); }

ImageView::ImageView(VkDevice device, VkImage image, VkFormat format,
                     VkImageAspectFlags aspectFlags, uint32_t mipLevels)
    : device{device} {
  VkImageViewCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  createInfo.image = image;
  createInfo.format = format;
  createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.subresourceRange.aspectMask = aspectFlags;
  createInfo.subresourceRange.baseMipLevel = 0;
  createInfo.subresourceRange.levelCount = mipLevels;
  createInfo.subresourceRange.baseArrayLayer = 0;
  createInfo.subresourceRange.layerCount = 1;

  throwOnError(
      [&]() { return vkCreateImageView(device, &createInfo, nullptr, &view); },
      "failed to create image view!");
}

ImageView::~ImageView() {
  if (view != nullptr)
    vkDestroyImageView(device, view, nullptr);
}

ImageView::ImageView(ImageView &&other) noexcept
    : device{other.device}, view{other.view} {
  other.view = nullptr;
}

ShaderModule::ShaderModule(VkDevice device, const std::vector<char> &code)
    : device{device} {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

  throwOnError(
      [&]() {
        return vkCreateShaderModule(device, &createInfo, nullptr, &module);
      },
      "failed to create shader module!");
}

ShaderModule::~ShaderModule() {
  vkDestroyShaderModule(device, module, nullptr);
}

PipelineLayout::PipelineLayout(
    VkDevice device, const std::vector<VkDescriptorSetLayout> &setLayouts,
    const std::vector<VkPushConstantRange> &pushConstantRanges)
    : device{device} {
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

  pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
  pipelineLayoutInfo.pSetLayouts = setLayouts.data();
  pipelineLayoutInfo.pushConstantRangeCount =
      static_cast<uint32_t>(pushConstantRanges.size());
  pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges.data();
  throwOnError(
      [&]() {
        return vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                                      &pipelineLayout);
      },
      "failed to create pipeline layout!");
}

PipelineLayout::~PipelineLayout() {
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

RenderPass::RenderPass(VkDevice device, VkPhysicalDevice physicalDevice,
                       VkSurfaceKHR surface)
    : device{device} {
  std::array<VkAttachmentDescription, 3> attachments{};

  attachments[0].format = swapSurfaceFormat(physicalDevice, surface).format;
  attachments[0].samples = maxUsableSampleCount(physicalDevice);
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  attachments[1].format = findDepthFormat(physicalDevice);
  attachments[1].samples = maxUsableSampleCount(physicalDevice);
  attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  attachments[2].format = swapSurfaceFormat(physicalDevice, surface).format;
  attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
  attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[2].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  std::array<VkAttachmentReference, 1> colorAttachmentRef{};
  colorAttachmentRef[0].attachment = 0;
  colorAttachmentRef[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthAttachmentRef{};
  depthAttachmentRef.attachment = 1;
  depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference colorAttachmentResolveRef{};
  colorAttachmentResolveRef.attachment = 2;
  colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  std::array<VkSubpassDescription, 1> subpass{};
  subpass[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass[0].colorAttachmentCount = colorAttachmentRef.size();
  subpass[0].pColorAttachments = colorAttachmentRef.data();
  subpass[0].pDepthStencilAttachment = &depthAttachmentRef;
  subpass[0].pResolveAttachments = &colorAttachmentResolveRef;

  std::array<VkSubpassDependency, 1> dependency{};
  dependency[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency[0].dstSubpass = 0;
  dependency[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency[0].srcAccessMask = 0;
  dependency[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
  renderPassInfo.pAttachments = attachments.data();
  renderPassInfo.subpassCount = subpass.size();
  renderPassInfo.pSubpasses = subpass.data();
  renderPassInfo.dependencyCount = dependency.size();
  renderPassInfo.pDependencies = dependency.data();

  throwOnError(
      [&]() {
        return vkCreateRenderPass(device, &renderPassInfo, nullptr,
                                  &renderPass);
      },
      "failed to create render pass!");
}

RenderPass::~RenderPass() { vkDestroyRenderPass(device, renderPass, nullptr); }

Pipeline::Pipeline(
    VkDevice device, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
    VkPipelineLayout pipelineLayout, VkRenderPass renderPass,
    const std::string &vertexShaderCodePath,
    const std::string &fragmentShaderCodePath, GLFWwindow *window,
    const std::vector<VkVertexInputAttributeDescription> &attributeDescriptions,
    const std::vector<VkVertexInputBindingDescription> &bindingDescription)
    : device{device} {
  const vulkan_wrappers::ShaderModule vertexShaderModule{
      device, readFile(vertexShaderCodePath)};
  const vulkan_wrappers::ShaderModule fragmentShaderModule{
      device, readFile(fragmentShaderCodePath)};

  std::array<VkGraphicsPipelineCreateInfo, 1> pipelineInfo{};
  pipelineInfo[0].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

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
  pipelineInfo[0].stageCount = shaderStages.size();
  pipelineInfo[0].pStages = shaderStages.data();

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  vertexInputInfo.vertexBindingDescriptionCount =
      static_cast<uint32_t>(bindingDescription.size());
  vertexInputInfo.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attributeDescriptions.size());
  vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data();
  vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  pipelineInfo[0].pVertexInputState = &vertexInputInfo;
  pipelineInfo[0].pInputAssemblyState = &inputAssembly;

  std::array<VkViewport, 1> viewport{};
  viewport[0].x = 0.0F;
  viewport[0].y = 0.0F;

  const auto swapChainExtent{swapExtent(physicalDevice, surface, window)};
  viewport[0].width = static_cast<float>(swapChainExtent.width);
  viewport[0].height = static_cast<float>(swapChainExtent.height);

  viewport[0].minDepth = 0.0F;
  viewport[0].maxDepth = 1.0F;

  std::array<VkRect2D, 1> scissor{};
  scissor[0].offset = {0, 0};
  scissor[0].extent = swapChainExtent;

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = viewport.size();
  viewportState.pViewports = viewport.data();
  viewportState.scissorCount = scissor.size();
  viewportState.pScissors = scissor.data();
  pipelineInfo[0].pViewportState = &viewportState;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0F;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;
  pipelineInfo[0].pRasterizationState = &rasterizer;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = maxUsableSampleCount(physicalDevice);
  pipelineInfo[0].pMultisampleState = &multisampling;

  std::array<VkPipelineColorBlendAttachmentState, 1> colorBlendAttachment{};
  colorBlendAttachment[0].colorWriteMask =
      static_cast<unsigned>(VK_COLOR_COMPONENT_R_BIT) |
      VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
      VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment[0].blendEnable = VK_FALSE;

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
  pipelineInfo[0].pColorBlendState = &colorBlending;

  pipelineInfo[0].layout = pipelineLayout;
  pipelineInfo[0].renderPass = renderPass;
  pipelineInfo[0].subpass = 0;
  pipelineInfo[0].basePipelineHandle = VK_NULL_HANDLE;

  VkPipelineDepthStencilStateCreateInfo depthStencil{};
  depthStencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;
  pipelineInfo[0].pDepthStencilState = &depthStencil;

  throwOnError(
      [&]() {
        return vkCreateGraphicsPipelines(
            device, VK_NULL_HANDLE, pipelineInfo.size(), pipelineInfo.data(),
            nullptr, &pipeline);
      },
      "failed to create graphics pipeline!");
}

Pipeline::~Pipeline() { vkDestroyPipeline(device, pipeline, nullptr); }

Framebuffer::Framebuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                         VkSurfaceKHR surface, VkRenderPass renderPass,
                         const std::vector<VkImageView> &attachments,
                         GLFWwindow *window)
    : device{device} {
  VkFramebufferCreateInfo framebufferInfo{};
  framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass = renderPass;
  framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
  framebufferInfo.pAttachments = attachments.data();

  const auto swapChainExtent{swapExtent(physicalDevice, surface, window)};
  framebufferInfo.width = swapChainExtent.width;
  framebufferInfo.height = swapChainExtent.height;

  framebufferInfo.layers = 1;

  throwOnError(
      [&]() {
        return vkCreateFramebuffer(device, &framebufferInfo, nullptr,
                                   &framebuffer);
      },
      "failed to create framebuffer!");
}

Framebuffer::~Framebuffer() {
  if (framebuffer != nullptr)
    vkDestroyFramebuffer(device, framebuffer, nullptr);
}

Framebuffer::Framebuffer(Framebuffer &&other) noexcept
    : device{other.device}, framebuffer{other.framebuffer} {
  other.framebuffer = nullptr;
}

CommandPool::CommandPool(VkDevice device, VkPhysicalDevice physicalDevice)
    : device{device} {
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex =
      graphicsSupportingQueueFamilyIndex(physicalDevice);
  throwOnError(
      [&]() {
        return vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
      },
      "failed to create command pool!");
}

CommandPool::~CommandPool() {
  vkDestroyCommandPool(device, commandPool, nullptr);
}

Semaphore::Semaphore(VkDevice device) : device{device} {
  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  throwOnError(
      [&]() {
        return vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore);
      },
      "failed to create semaphore!");
}

Semaphore::~Semaphore() {
  if (semaphore != nullptr)
    vkDestroySemaphore(device, semaphore, nullptr);
}

Semaphore::Semaphore(Semaphore &&other) noexcept
    : device{other.device}, semaphore{other.semaphore} {
  other.semaphore = nullptr;
}

Fence::Fence(VkDevice device, VkFenceCreateFlags flags) : device{device} {
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = flags;
  throwOnError(
      [&]() { return vkCreateFence(device, &fenceInfo, nullptr, &fence); },
      "failed to create fence!");
}

Fence::~Fence() {
  if (fence != nullptr)
    vkDestroyFence(device, fence, nullptr);
}

Fence::Fence(Fence &&other) noexcept
    : device{other.device}, fence{other.fence} {
  other.fence = nullptr;
}

CommandBuffers::CommandBuffers(VkDevice device, VkCommandPool commandPool,
                               std::vector<VkCommandBuffer>::size_type size)
    : device{device}, commandPool{commandPool}, commandBuffers(size) {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;

  allocInfo.commandPool = commandPool;

  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
  throwOnError(
      [&]() {
        return vkAllocateCommandBuffers(device, &allocInfo,
                                        commandBuffers.data());
      },
      "failed to allocate command buffers!");
}

CommandBuffers::~CommandBuffers() {
  vkFreeCommandBuffers(device, commandPool,
                       static_cast<uint32_t>(commandBuffers.size()),
                       commandBuffers.data());
}

Buffer::Buffer(VkDevice device, VkBufferUsageFlags usage,
               VkDeviceSize bufferSize)
    : device{device} {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = bufferSize;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  throwOnError(
      [&]() { return vkCreateBuffer(device, &bufferInfo, nullptr, &buffer); },
      "failed to create buffer!");
}

Buffer::~Buffer() {
  if (buffer != nullptr)
    vkDestroyBuffer(device, buffer, nullptr);
}

Buffer::Buffer(Buffer &&other) noexcept
    : device{other.device}, buffer{other.buffer} {
  other.buffer = nullptr;
}

DeviceMemory::DeviceMemory(VkDevice device, VkPhysicalDevice physicalDevice,
                           VkMemoryPropertyFlags properties,
                           const VkMemoryRequirements &memoryRequirements)
    : device{device} {
  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memoryRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      physicalDevice, memoryRequirements.memoryTypeBits, properties);

  throwOnError(
      [&]() { return vkAllocateMemory(device, &allocInfo, nullptr, &memory); },
      "failed to allocate device memory!");
}

DeviceMemory::~DeviceMemory() {
  if (memory != nullptr)
    vkFreeMemory(device, memory, nullptr);
}

DeviceMemory::DeviceMemory(DeviceMemory &&other) noexcept
    : device{other.device}, memory{other.memory} {
  other.memory = nullptr;
}

DescriptorSetLayout::DescriptorSetLayout(
    VkDevice device, const std::vector<VkDescriptorSetLayoutBinding> &bindings)
    : device{device} {
  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutInfo.pBindings = bindings.data();

  throwOnError(
      [&]() {
        return vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                           &descriptorSetLayout);
      },
      "failed to create descriptor set layout!");
}

DescriptorSetLayout::~DescriptorSetLayout() {
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

DescriptorPool::DescriptorPool(
    VkDevice device, const std::vector<VkDescriptorPoolSize> &poolSize,
    uint32_t maxSets)
    : device{device} {
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSize.size());
  poolInfo.pPoolSizes = poolSize.data();
  poolInfo.maxSets = maxSets;

  throwOnError(
      [&]() {
        return vkCreateDescriptorPool(device, &poolInfo, nullptr,
                                      &descriptorPool);
      },
      "failed to create descriptor pool!");
}

DescriptorPool::~DescriptorPool() {
  if (descriptorPool != nullptr)
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
}

DescriptorPool::DescriptorPool(DescriptorPool &&other) noexcept
    : device{other.device}, descriptorPool{other.descriptorPool} {
  other.descriptorPool = nullptr;
}

Image::Image(VkDevice device, uint32_t width, uint32_t height, VkFormat format,
             VkImageTiling tiling, VkImageUsageFlags usage, uint32_t mipLevels,
             VkSampleCountFlagBits samples)
    : device{device} {
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = mipLevels;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = samples;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  throwOnError(
      [&]() { return vkCreateImage(device, &imageInfo, nullptr, &image); },
      "failed to create image!");
}

Image::~Image() {
  if (image != nullptr)
    vkDestroyImage(device, image, nullptr);
}

Image::Image(Image &&other) noexcept
    : device{other.device}, image{other.image} {
  other.image = nullptr;
}

Sampler::Sampler(VkDevice device, VkPhysicalDevice physicalDevice,
                 uint32_t mipLevels)
    : device{device} {
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.anisotropyEnable = VK_TRUE;

  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physicalDevice, &properties);
  samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.mipLodBias = 0.0F;
  samplerInfo.minLod = 0.0F;
  samplerInfo.maxLod = static_cast<float>(mipLevels);

  throwOnError(
      [&]() {
        return vkCreateSampler(device, &samplerInfo, nullptr, &sampler);
      },
      "failed to create texture sampler!");
}

Sampler::~Sampler() { vkDestroySampler(device, sampler, nullptr); }
} // namespace vulkan_wrappers

static auto supportsGraphics(const VkQueueFamilyProperties &properties)
    -> bool {
  return (properties.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U;
}

auto queueFamilyPropertiesCount(VkPhysicalDevice device) -> uint32_t {
  return vulkanCountFromPhysicalDevice(
      device, [](VkPhysicalDevice device_, uint32_t *count) {
        vkGetPhysicalDeviceQueueFamilyProperties(device_, count, nullptr);
      });
}

auto queueFamilyProperties(VkPhysicalDevice device, uint32_t count)
    -> std::vector<VkQueueFamilyProperties> {
  std::vector<VkQueueFamilyProperties> properties(count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, properties.data());
  return properties;
}

auto graphicsSupportingQueueFamilyIndex(
    const std::vector<VkQueueFamilyProperties> &queueFamilyProperties)
    -> uint32_t {
  return static_cast<uint32_t>(std::distance(
      queueFamilyProperties.begin(),
      std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
                   supportsGraphics)));
}

auto graphicsSupportingQueueFamilyIndex(VkPhysicalDevice device) -> uint32_t {
  return graphicsSupportingQueueFamilyIndex(
      queueFamilyProperties(device, queueFamilyPropertiesCount(device)));
}

auto presentSupportingQueueFamilyIndex(VkPhysicalDevice device,
                                       VkSurfaceKHR surface) -> uint32_t {
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

auto swapExtent(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
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

auto findDepthFormat(VkPhysicalDevice physicalDevice) -> VkFormat {
  for (const auto format : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT,
                            VK_FORMAT_D16_UNORM}) {
    VkFormatProperties properties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
    if ((properties.optimalTilingFeatures &
         VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) ==
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
      return format;
  }
  throw std::runtime_error("failed to find depth format");
}

void throwOnError(const std::function<VkResult()> &f,
                  const std::string &message) {
  if (f() != VK_SUCCESS)
    throw std::runtime_error{message};
}

auto bufferMemory(VkDevice device, VkPhysicalDevice physicalDevice,
                  VkBuffer buffer, VkMemoryPropertyFlags flags)
    -> vulkan_wrappers::DeviceMemory {
  VkMemoryRequirements memoryRequirements;
  vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
  vulkan_wrappers::DeviceMemory memory{device, physicalDevice, flags,
                                       memoryRequirements};
  vkBindBufferMemory(device, buffer, memory.memory, 0);
  return memory;
}

auto imageMemory(VkDevice device, VkPhysicalDevice physicalDevice,
                 VkImage image, VkMemoryPropertyFlags flags)
    -> vulkan_wrappers::DeviceMemory {
  VkMemoryRequirements memoryRequirements;
  vkGetImageMemoryRequirements(device, image, &memoryRequirements);
  vulkan_wrappers::DeviceMemory memory{device, physicalDevice, flags,
                                       memoryRequirements};
  vkBindImageMemory(device, image, memory.memory, 0);
  return memory;
}

static void begin(VkCommandBuffer buffer) {
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(buffer, &beginInfo);
}

static void
submitAndWait(const vulkan_wrappers::CommandBuffers &vulkanCommandBuffers,
              VkQueue graphicsQueue) {
  std::array<VkSubmitInfo, 1> submitInfo{};
  submitInfo[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo[0].commandBufferCount =
      static_cast<uint32_t>(vulkanCommandBuffers.commandBuffers.size());
  submitInfo[0].pCommandBuffers = vulkanCommandBuffers.commandBuffers.data();
  vkQueueSubmit(graphicsQueue, submitInfo.size(), submitInfo.data(),
                VK_NULL_HANDLE);
  vkQueueWaitIdle(graphicsQueue);
}

static void copyBuffer(VkDevice device, VkCommandPool commandPool,
                       VkQueue graphicsQueue, VkBuffer sourceBuffer,
                       VkBuffer destinationBuffer, VkDeviceSize size) {
  vulkan_wrappers::CommandBuffers vulkanCommandBuffers{device, commandPool, 1};
  begin(vulkanCommandBuffers.commandBuffers[0]);
  std::array<VkBufferCopy, 1> copyRegion{};
  copyRegion[0].size = size;
  vkCmdCopyBuffer(vulkanCommandBuffers.commandBuffers[0], sourceBuffer,
                  destinationBuffer, copyRegion.size(), copyRegion.data());
  vkEndCommandBuffer(vulkanCommandBuffers.commandBuffers[0]);
  submitAndWait(vulkanCommandBuffers, graphicsQueue);
}

static void copy(VkDevice device, VkPhysicalDevice physicalDevice,
                 VkCommandPool commandPool, VkQueue graphicsQueue,
                 VkBuffer destinationBuffer, const void *source, size_t size) {
  const vulkan_wrappers::Buffer stagingBuffer{
      device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, size};
  const auto memory{bufferMemory(device, physicalDevice, stagingBuffer.buffer,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
  copy(device, memory.memory, source, size);
  copyBuffer(device, commandPool, graphicsQueue, stagingBuffer.buffer,
             destinationBuffer, size);
}

void copyBufferToImage(VkDevice device, VkCommandPool commandPool,
                       VkQueue graphicsQueue, VkBuffer buffer, VkImage image,
                       uint32_t width, uint32_t height) {
  vulkan_wrappers::CommandBuffers vulkanCommandBuffers{device, commandPool, 1};
  begin(vulkanCommandBuffers.commandBuffers[0]);

  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {width, height, 1};

  vkCmdCopyBufferToImage(vulkanCommandBuffers.commandBuffers[0], buffer, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  vkEndCommandBuffer(vulkanCommandBuffers.commandBuffers[0]);
  submitAndWait(vulkanCommandBuffers, graphicsQueue);
}

void copy(VkDevice device, VkDeviceMemory memory, const void *source,
          size_t size) {
  void *data = nullptr;
  vkMapMemory(device, memory, 0, size, 0, &data);
  memcpy(data, source, size);
  vkUnmapMemory(device, memory);
}

auto swapChainImages(VkDevice device, VkSwapchainKHR swapChain)
    -> std::vector<VkImage> {
  uint32_t imageCount{0};
  vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
  std::vector<VkImage> swapChainImages(imageCount);
  vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
                          swapChainImages.data());
  return swapChainImages;
}

static auto vulkanDevices(VkInstance instance)
    -> std::vector<VkPhysicalDevice> {
  uint32_t deviceCount{0};
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
  return devices;
}

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
  transform(extensionProperties.begin(), extensionProperties.end(),
            extensionNames.begin(),
            [](const VkExtensionProperties &properties) {
              return static_cast<const char *>(properties.extensionName);
            });

  if (!includes(extensionNames.begin(), extensionNames.end(),
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

  VkPhysicalDeviceFeatures supportedFeatures;
  vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

  if (supportedFeatures.samplerAnisotropy == 0U)
    return false;

  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(device, &deviceProperties);
  if (deviceProperties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
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

auto suitableDevice(VkInstance instance, VkSurfaceKHR surface)
    -> VkPhysicalDevice {
  for (auto *const device : vulkanDevices(instance))
    if (suitable(device, surface))
      return device;
  throw std::runtime_error("failed to find a suitable GPU!");
}

void transitionImageLayout(VkDevice device, VkCommandPool commandPool,
                           VkQueue graphicsQueue, VkImage image,
                           VkImageLayout oldLayout, VkImageLayout newLayout,
                           uint32_t mipLevels) {
  vulkan_wrappers::CommandBuffers vulkanCommandBuffers{device, commandPool, 1};
  begin(vulkanCommandBuffers.commandBuffers[0]);

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = mipLevels;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags sourceStage{0};
  VkPipelineStageFlags destinationStage{0};

  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
      newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else
    throw std::invalid_argument("unsupported layout transition!");

  vkCmdPipelineBarrier(vulkanCommandBuffers.commandBuffers[0], sourceStage,
                       destinationStage, 0, 0, nullptr, 0, nullptr, 1,
                       &barrier);
  vkEndCommandBuffer(vulkanCommandBuffers.commandBuffers[0]);
  submitAndWait(vulkanCommandBuffers, graphicsQueue);
}

void generateMipmaps(VkDevice device, VkPhysicalDevice physicalDevice,
                     VkCommandPool commandPool, VkQueue graphicsQueue,
                     VkImage image, VkFormat imageFormat, int32_t texWidth,
                     int32_t texHeight, uint32_t mipLevels) {
  VkFormatProperties formatProperties;
  vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat,
                                      &formatProperties);

  if ((formatProperties.optimalTilingFeatures &
       VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) == 0U)
    throw std::runtime_error(
        "texture image format does not support linear blitting!");

  vulkan_wrappers::CommandBuffers vulkanCommandBuffers{device, commandPool, 1};
  begin(vulkanCommandBuffers.commandBuffers[0]);

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.image = image;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.subresourceRange.levelCount = 1;

  auto mipWidth{texWidth};
  auto mipHeight{texHeight};

  for (auto i{1U}; i < mipLevels; i++) {
    barrier.subresourceRange.baseMipLevel = i - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        vulkanCommandBuffers.commandBuffers[0], VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkImageBlit blit{};
    blit.srcOffsets[0] = {0, 0, 0};
    blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.mipLevel = i - 1;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;
    blit.dstOffsets[0] = {0, 0, 0};
    blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1,
                          mipHeight > 1 ? mipHeight / 2 : 1, 1};
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.mipLevel = i;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = 1;

    vkCmdBlitImage(vulkanCommandBuffers.commandBuffers[0], image,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                   VK_FILTER_LINEAR);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(vulkanCommandBuffers.commandBuffers[0],
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);

    if (mipWidth > 1)
      mipWidth /= 2;
    if (mipHeight > 1)
      mipHeight /= 2;
  }

  barrier.subresourceRange.baseMipLevel = mipLevels - 1;
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(vulkanCommandBuffers.commandBuffers[0],
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  vkEndCommandBuffer(vulkanCommandBuffers.commandBuffers[0]);
  submitAndWait(vulkanCommandBuffers, graphicsQueue);
}

auto swapChainImageViews(VkDevice device, VkPhysicalDevice physicalDevice,
                         VkSurfaceKHR surface,
                         const std::vector<VkImage> &swapChainImages)
    -> std::vector<vulkan_wrappers::ImageView> {
  const auto format{swapSurfaceFormat(physicalDevice, surface).format};
  std::vector<vulkan_wrappers::ImageView> swapChainImageViews;
  transform(swapChainImages.begin(), swapChainImages.end(),
            back_inserter(swapChainImageViews),
            [device, format](VkImage image) {
              return vulkan_wrappers::ImageView{device, image, format,
                                                VK_IMAGE_ASPECT_COLOR_BIT, 1};
            });
  return swapChainImageViews;
}

auto descriptor(
    const vulkan_wrappers::Device &vulkanDevice,
    const vulkan_wrappers::ImageView &vulkanTextureImageView,
    const vulkan_wrappers::Sampler &vulkanTextureSampler,
    const vulkan_wrappers::DescriptorSetLayout &vulkanDescriptorSetLayout,
    const std::vector<VkImage> &swapChainImages,
    const std::vector<VulkanBufferWithMemory> &vulkanUniformBuffers,
    VkDeviceSize bufferObjectSize) -> VulkanDescriptor {

  std::vector<VkDescriptorPoolSize> poolSize(2);
  poolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSize[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
  poolSize[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSize[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

  vulkan_wrappers::DescriptorPool vulkanDescriptorPool{
      vulkanDevice.device, poolSize,
      static_cast<uint32_t>(swapChainImages.size())};

  std::vector<VkDescriptorSet> descriptorSets(swapChainImages.size());
  std::vector<VkDescriptorSetLayout> layouts(
      swapChainImages.size(), vulkanDescriptorSetLayout.descriptorSetLayout);

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = vulkanDescriptorPool.descriptorPool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
  allocInfo.pSetLayouts = layouts.data();

  throwOnError(
      [&]() {
        // no deallocate needed here per tutorial
        return vkAllocateDescriptorSets(vulkanDevice.device, &allocInfo,
                                        descriptorSets.data());
      },
      "failed to allocate descriptor sets!");

  for (auto i{0U}; i < swapChainImages.size(); i++) {
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = vulkanUniformBuffers[i].buffer.buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = bufferObjectSize;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = vulkanTextureImageView.view;
    imageInfo.sampler = vulkanTextureSampler.sampler;

    std::array<VkWriteDescriptorSet, 2> descriptorWrite{};
    descriptorWrite[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite[0].dstSet = descriptorSets[i];
    descriptorWrite[0].dstBinding = 0;
    descriptorWrite[0].dstArrayElement = 0;
    descriptorWrite[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite[0].descriptorCount = 1;
    descriptorWrite[0].pBufferInfo = &bufferInfo;

    descriptorWrite[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite[1].dstSet = descriptorSets[i];
    descriptorWrite[1].dstBinding = 1;
    descriptorWrite[1].dstArrayElement = 0;
    descriptorWrite[1].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrite[1].descriptorCount = 1;
    descriptorWrite[1].pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(vulkanDevice.device, descriptorWrite.size(),
                           descriptorWrite.data(), 0, nullptr);
  }
  return VulkanDescriptor{std::move(vulkanDescriptorPool),
                          std::move(descriptorSets)};
}

auto present(VkQueue presentQueue, const vulkan_wrappers::Swapchain &swapchain,
             uint32_t imageIndex, VkSemaphore signalSemaphore) -> VkResult {
  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

  std::array<VkSemaphore, 1> signalSemaphores{signalSemaphore};
  presentInfo.waitSemaphoreCount = signalSemaphores.size();
  presentInfo.pWaitSemaphores = signalSemaphores.data();

  const std::array<VkSwapchainKHR, 1> swapChains = {swapchain.swapChain};
  presentInfo.swapchainCount = swapChains.size();
  presentInfo.pSwapchains = swapChains.data();

  presentInfo.pImageIndices = &imageIndex;

  return vkQueuePresentKHR(presentQueue, &presentInfo);
}

void submit(const vulkan_wrappers::Device &device, VkQueue graphicsQueue,
            VkSemaphore imageAvailableSemaphore,
            VkSemaphore renderFinishedSemaphore, VkFence inFlightFence,
            VkCommandBuffer commandBuffer) {
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  const std::array<VkSemaphore, 1> waitSemaphores = {imageAvailableSemaphore};
  const std::array<VkPipelineStageFlags, 1> waitStages = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.waitSemaphoreCount = waitSemaphores.size();
  submitInfo.pWaitSemaphores = waitSemaphores.data();
  submitInfo.pWaitDstStageMask = waitStages.data();

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  const std::array<VkSemaphore, 1> signalSemaphores = {renderFinishedSemaphore};
  submitInfo.signalSemaphoreCount = signalSemaphores.size();
  submitInfo.pSignalSemaphores = signalSemaphores.data();

  vkResetFences(device.device, 1, &inFlightFence);
  throwOnError(
      [&]() {
        return vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence);
      },
      "failed to submit draw command buffer!");
}

auto frameImage(VkDevice device, VkPhysicalDevice physicalDevice,
                VkSurfaceKHR surface, GLFWwindow *window, VkFormat format,
                VkImageUsageFlags usageFlags, VkImageAspectFlags aspectFlags)
    -> VulkanImage {
  const auto swapChainExtent{swapExtent(physicalDevice, surface, window)};
  vulkan_wrappers::Image image{device,
                               swapChainExtent.width,
                               swapChainExtent.height,
                               format,
                               VK_IMAGE_TILING_OPTIMAL,
                               usageFlags,
                               1,
                               maxUsableSampleCount(physicalDevice)};
  auto memory{imageMemory(device, physicalDevice, image.image,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
  vulkan_wrappers::ImageView view{device, image.image, format, aspectFlags, 1};
  return VulkanImage{std::move(image), std::move(view), std::move(memory)};
}

auto bufferWithMemory(VkDevice device, VkPhysicalDevice physicalDevice,
                      VkCommandPool commandPool, VkQueue graphicsQueue,
                      VkBufferUsageFlags usageFlags, const void *source,
                      size_t size) -> VulkanBufferWithMemory {
  vulkan_wrappers::Buffer buffer{device, usageFlags, size};
  auto memory{bufferMemory(device, physicalDevice, buffer.buffer,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
  copy(device, physicalDevice, commandPool, graphicsQueue, buffer.buffer,
       source, size);
  return VulkanBufferWithMemory{std::move(buffer), std::move(memory)};
}

auto semaphores(VkDevice device, int n)
    -> std::vector<vulkan_wrappers::Semaphore> {
  std::vector<vulkan_wrappers::Semaphore> semaphores;
  generate_n(back_inserter(semaphores), n,
             [device]() { return vulkan_wrappers::Semaphore{device}; });
  return semaphores;
}

void throwIfFailsToBegin(VkCommandBuffer commandBuffer) {
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  throwOnError(
      [&]() { return vkBeginCommandBuffer(commandBuffer, &beginInfo); },
      "failed to begin recording command buffer!");
}

void beginRenderPass(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
                     VkFramebuffer framebuffer, VkCommandBuffer commandBuffer,
                     GLFWwindow *window, VkRenderPass renderPass) {
  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = renderPass;
  renderPassInfo.framebuffer = framebuffer;
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent =
      swapExtent(physicalDevice, surface, window);

  std::array<VkClearValue, 2> clearValues{};
  clearValues[0].color = {{0.F, 0.F, 0.F, 1.F}};
  clearValues[1].depthStencil = {1.F, 0};
  renderPassInfo.clearValueCount = clearValues.size();
  renderPassInfo.pClearValues = clearValues.data();

  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);
}
} // namespace sbash64::graphics
