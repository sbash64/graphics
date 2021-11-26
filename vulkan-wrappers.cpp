#include <cstddef>
#include <sbash64/graphics/load-object.hpp>
#include <sbash64/graphics/vulkan-wrappers.hpp>

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>

namespace sbash64::graphics {
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

PipelineLayout::PipelineLayout(VkDevice device,
                               VkDescriptorSetLayout descriptorSetLayout)
    : device{device} {
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

  const std::array<VkDescriptorSetLayout, 1> descriptorSetLayouts{
      descriptorSetLayout};
  pipelineLayoutInfo.setLayoutCount = descriptorSetLayouts.size();
  pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
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

Pipeline::Pipeline(VkDevice device, VkPhysicalDevice physicalDevice,
                   VkSurfaceKHR surface, VkPipelineLayout pipelineLayout,
                   VkRenderPass renderPass,
                   const std::string &vertexShaderCodePath,
                   const std::string &fragmentShaderCodePath,
                   GLFWwindow *window)
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

  std::array<VkVertexInputBindingDescription, 1> bindingDescription{};
  bindingDescription[0].binding = 0;
  bindingDescription[0].stride = sizeof(Vertex);
  bindingDescription[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(Vertex, position);

  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(Vertex, textureCoordinate);

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

Fence::Fence(VkDevice device) : device{device} {
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
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

DescriptorSetLayout::DescriptorSetLayout(VkDevice device) : device{device} {
  VkDescriptorSetLayoutBinding uboLayoutBinding{};
  uboLayoutBinding.binding = 0;
  uboLayoutBinding.descriptorCount = 1;
  uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uboLayoutBinding.pImmutableSamplers = nullptr;
  uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkDescriptorSetLayoutBinding samplerLayoutBinding{};
  samplerLayoutBinding.binding = 1;
  samplerLayoutBinding.descriptorCount = 1;
  samplerLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  samplerLayoutBinding.pImmutableSamplers = nullptr;
  samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding,
                                                          samplerLayoutBinding};
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

DescriptorPool::DescriptorPool(VkDevice device,
                               const std::vector<VkImage> &swapChainImages)
    : device{device} {
  std::array<VkDescriptorPoolSize, 2> poolSize{};
  poolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSize[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
  poolSize[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSize[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = poolSize.size();
  poolInfo.pPoolSizes = poolSize.data();
  poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

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

auto supportsGraphics(const VkQueueFamilyProperties &properties) -> bool {
  return (properties.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U;
}

auto vulkanCountFromPhysicalDevice(
    VkPhysicalDevice device,
    const std::function<void(VkPhysicalDevice, uint32_t *)> &f) -> uint32_t {
  uint32_t count{0};
  f(device, &count);
  return count;
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

auto maxUsableSampleCount(VkPhysicalDevice physicalDevice)
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
} // namespace sbash64::graphics
