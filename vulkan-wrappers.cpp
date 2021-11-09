#include <sbash64/graphics/vulkan-wrappers.hpp>

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>

namespace sbash64::graphics {
auto swapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats)
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
  applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  applicationInfo.pEngineName = "N/A";
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

  VkPhysicalDeviceFeatures deviceFeatures{};
  deviceFeatures.samplerAnisotropy = VK_TRUE;

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

Device::~Device() { vkDestroyDevice(device, nullptr); }

Surface::Surface(VkInstance instance, GLFWwindow *window) : instance{instance} {
  if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
      VK_SUCCESS)
    throw std::runtime_error("failed to create window surface!");
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

  auto imageCount{capabilities.minImageCount + 1};
  if (capabilities.maxImageCount > 0 &&
      imageCount > capabilities.maxImageCount) {
    imageCount = capabilities.maxImageCount;
  }

  auto formatCount{vulkanCountFromPhysicalDevice(
      physicalDevice, [&surface](VkPhysicalDevice device_, uint32_t *count) {
        vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, count, nullptr);
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

Swapchain::~Swapchain() { vkDestroySwapchainKHR(device, swapChain, nullptr); }

ImageView::ImageView(VkDevice device, VkImage image, VkFormat format)
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
  createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  createInfo.subresourceRange.baseMipLevel = 0;
  createInfo.subresourceRange.levelCount = 1;
  createInfo.subresourceRange.baseArrayLayer = 0;
  createInfo.subresourceRange.layerCount = 1;

  if (vkCreateImageView(device, &createInfo, nullptr, &view) != VK_SUCCESS)
    throw std::runtime_error("failed to create image view!");
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

  if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS)
    throw std::runtime_error("failed to create shader module!");
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
  if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                             &pipelineLayout) != VK_SUCCESS)
    throw std::runtime_error("failed to create pipeline layout!");
}

PipelineLayout::~PipelineLayout() {
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

RenderPass::RenderPass(VkDevice device, VkPhysicalDevice physicalDevice,
                       VkSurfaceKHR surface)
    : device{device} {
  std::array<VkAttachmentDescription, 1> colorAttachment{};

  auto formatCount{vulkanCountFromPhysicalDevice(
      physicalDevice, [&surface](VkPhysicalDevice device_, uint32_t *count) {
        vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, count, nullptr);
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
  pipelineInfo.at(0).sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

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
  pipelineInfo.at(0).stageCount = shaderStages.size();
  pipelineInfo.at(0).pStages = shaderStages.data();

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

  pipelineInfo.at(0).pVertexInputState = &vertexInputInfo;
  pipelineInfo.at(0).pInputAssemblyState = &inputAssembly;

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
  pipelineInfo.at(0).pViewportState = &viewportState;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0F;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;
  pipelineInfo.at(0).pRasterizationState = &rasterizer;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  pipelineInfo.at(0).pMultisampleState = &multisampling;

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
  pipelineInfo.at(0).pColorBlendState = &colorBlending;

  pipelineInfo.at(0).layout = pipelineLayout;
  pipelineInfo.at(0).renderPass = renderPass;
  pipelineInfo.at(0).subpass = 0;
  pipelineInfo.at(0).basePipelineHandle = VK_NULL_HANDLE;

  if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, pipelineInfo.size(),
                                pipelineInfo.data(), nullptr,
                                &pipeline) != VK_SUCCESS)
    throw std::runtime_error("failed to create graphics pipeline!");
}

Pipeline::~Pipeline() { vkDestroyPipeline(device, pipeline, nullptr); }

Framebuffer::Framebuffer(VkDevice device, VkPhysicalDevice physicalDevice,
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
  if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
      VK_SUCCESS)
    throw std::runtime_error("failed to create command pool!");
}

CommandPool::~CommandPool() {
  vkDestroyCommandPool(device, commandPool, nullptr);
}

Semaphore::Semaphore(VkDevice device) : device{device} {
  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) !=
      VK_SUCCESS)
    throw std::runtime_error("failed to create semaphore!");
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
  if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
    throw std::runtime_error("failed to create fence!");
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
  if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) !=
      VK_SUCCESS)
    throw std::runtime_error("failed to allocate command buffers!");
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
  if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
    throw std::runtime_error("failed to create buffer!");
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

  if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
    throw std::runtime_error("failed to allocate device memory!");
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

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &uboLayoutBinding;

  if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                  &descriptorSetLayout) != VK_SUCCESS)
    throw std::runtime_error("failed to create descriptor set layout!");
}

DescriptorSetLayout::~DescriptorSetLayout() {
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

DescriptorPool::DescriptorPool(VkDevice device,
                               const std::vector<VkImage> &swapChainImages)
    : device{device} {
  std::array<VkDescriptorPoolSize, 1> poolSize{};
  poolSize.at(0).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSize.at(0).descriptorCount =
      static_cast<uint32_t>(swapChainImages.size());

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = poolSize.size();
  poolInfo.pPoolSizes = poolSize.data();
  poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

  if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) !=
      VK_SUCCESS)
    throw std::runtime_error("failed to create descriptor pool!");
}

DescriptorPool::~DescriptorPool() {
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
}
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
} // namespace sbash64::graphics
