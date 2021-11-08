#ifndef SBASH64_GRAPHICS_VULKAN_WRAPPERS_HPP_
#define SBASH64_GRAPHICS_VULKAN_WRAPPERS_HPP_

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <array>
#include <functional>
#include <vector>

namespace sbash64::graphics {
struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;
};

constexpr std::array<const char *, 1> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

auto supportsGraphics(const VkQueueFamilyProperties &properties) -> bool;
auto vulkanCountFromPhysicalDevice(
    VkPhysicalDevice device,
    const std::function<void(VkPhysicalDevice, uint32_t *)> &f) -> uint32_t;
auto queueFamilyPropertiesCount(VkPhysicalDevice device) -> uint32_t;
auto queueFamilyProperties(VkPhysicalDevice device, uint32_t count)
    -> std::vector<VkQueueFamilyProperties>;
auto graphicsSupportingQueueFamilyIndex(
    const std::vector<VkQueueFamilyProperties> &queueFamilyProperties)
    -> uint32_t;
auto graphicsSupportingQueueFamilyIndex(VkPhysicalDevice device) -> uint32_t;
auto presentSupportingQueueFamilyIndex(VkPhysicalDevice device,
                                       VkSurfaceKHR surface) -> uint32_t;
auto swapExtent(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
                GLFWwindow *window) -> VkExtent2D;

namespace vulkan_wrappers {
struct Instance {
  Instance();
  ~Instance();

  Instance(const Instance &) = delete;
  auto operator=(const Instance &) -> Instance & = delete;
  Instance(Instance &&) = delete;
  auto operator=(Instance &&) -> Instance & = delete;

  VkInstance instance{};
};

struct Device {
  Device(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
  ~Device();

  Device(const Device &) = delete;
  auto operator=(const Device &) -> Device & = delete;
  Device(Device &&) = delete;
  auto operator=(Device &&) -> Device & = delete;

  VkDevice device{};
};

struct Surface {
  Surface(VkInstance instance, GLFWwindow *window);
  ~Surface();

  Surface(const Surface &) = delete;
  auto operator=(const Surface &) -> Surface & = delete;
  Surface(Surface &&) = delete;
  auto operator=(Surface &&) -> Surface & = delete;

  VkInstance instance;
  VkSurfaceKHR surface{};
};

struct Swapchain {
  Swapchain(VkDevice device, VkPhysicalDevice physicalDevice,
            VkSurfaceKHR surface, GLFWwindow *window);
  ~Swapchain();

  Swapchain(const Swapchain &) = delete;
  auto operator=(const Swapchain &) -> Swapchain & = delete;
  Swapchain(Swapchain &&) = delete;
  auto operator=(Swapchain &&) -> Swapchain & = delete;

  VkDevice device;
  VkSwapchainKHR swapChain{};
};

struct ImageView {
  ImageView(VkDevice device, VkPhysicalDevice physicalDevice,
            VkSurfaceKHR surface, VkImage image);
  ~ImageView();
  ImageView(ImageView &&) noexcept;

  auto operator=(ImageView &&) -> ImageView & = delete;
  ImageView(const ImageView &) = delete;
  auto operator=(const ImageView &) -> ImageView & = delete;

  VkDevice device;
  VkImageView view{};
};

struct ShaderModule {
  ShaderModule(VkDevice device, const std::vector<char> &code);
  ~ShaderModule();

  ShaderModule(ShaderModule &&) = delete;
  auto operator=(ShaderModule &&) -> ShaderModule & = delete;
  ShaderModule(const ShaderModule &) = delete;
  auto operator=(const ShaderModule &) -> ShaderModule & = delete;

  VkDevice device;
  VkShaderModule module{};
};

struct PipelineLayout {
  PipelineLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout);
  ~PipelineLayout();

  PipelineLayout(PipelineLayout &&) = delete;
  auto operator=(PipelineLayout &&) -> PipelineLayout & = delete;
  PipelineLayout(const PipelineLayout &) = delete;
  auto operator=(const PipelineLayout &) -> PipelineLayout & = delete;

  VkDevice device;
  VkPipelineLayout pipelineLayout{};
};

struct RenderPass {
  RenderPass(VkDevice device, VkPhysicalDevice physicalDevice,
             VkSurfaceKHR surface);
  ~RenderPass();

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
           const std::string &fragmentShaderCodePath, GLFWwindow *window);
  ~Pipeline();

  Pipeline(Pipeline &&) = delete;
  auto operator=(Pipeline &&) -> Pipeline & = delete;
  Pipeline(const Pipeline &) = delete;
  auto operator=(const Pipeline &) -> Pipeline & = delete;

  VkDevice device;
  VkPipeline pipeline{};
};

struct Framebuffer {
  Framebuffer(VkDevice, VkPhysicalDevice, VkSurfaceKHR, VkRenderPass,
              VkImageView imageView, GLFWwindow *window);
  ~Framebuffer();
  Framebuffer(Framebuffer &&) noexcept;

  auto operator=(Framebuffer &&) -> Framebuffer & = delete;
  Framebuffer(const Framebuffer &) = delete;
  auto operator=(const Framebuffer &) -> Framebuffer & = delete;

  VkDevice device;
  VkFramebuffer framebuffer{};
};

struct CommandPool {
  CommandPool(VkDevice, VkPhysicalDevice);
  ~CommandPool();

  CommandPool(CommandPool &&) = delete;
  auto operator=(CommandPool &&) -> CommandPool & = delete;
  CommandPool(const CommandPool &) = delete;
  auto operator=(const CommandPool &) -> CommandPool & = delete;

  VkDevice device;
  VkCommandPool commandPool{};
};

struct Semaphore {
  explicit Semaphore(VkDevice);
  ~Semaphore();
  Semaphore(Semaphore &&) noexcept;

  auto operator=(Semaphore &&) -> Semaphore & = delete;
  Semaphore(const Semaphore &) = delete;
  auto operator=(const Semaphore &) -> Semaphore & = delete;

  VkDevice device;
  VkSemaphore semaphore{};
};

struct Fence {
  explicit Fence(VkDevice);
  ~Fence();
  Fence(Fence &&) noexcept;

  auto operator=(Fence &&) -> Fence & = delete;
  Fence(const Fence &) = delete;
  auto operator=(const Fence &) -> Fence & = delete;

  VkDevice device;
  VkFence fence{};
};

struct CommandBuffers {
  CommandBuffers(VkDevice, VkCommandPool,
                 std::vector<VkCommandBuffer>::size_type size);
  ~CommandBuffers();

  CommandBuffers(CommandBuffers &&) = delete;
  auto operator=(CommandBuffers &&) -> CommandBuffers & = delete;
  CommandBuffers(const CommandBuffers &) = delete;
  auto operator=(const CommandBuffers &) -> CommandBuffers & = delete;

  VkDevice device;
  VkCommandPool commandPool;
  std::vector<VkCommandBuffer> commandBuffers;
};

struct Buffer {
  Buffer(VkDevice, VkBufferUsageFlags usage, VkDeviceSize bufferSize);
  ~Buffer();
  Buffer(Buffer &&) noexcept;

  auto operator=(Buffer &&) -> Buffer & = delete;
  Buffer(const Buffer &) = delete;
  auto operator=(const Buffer &) -> Buffer & = delete;

  VkDevice device;
  VkBuffer buffer{};
};

struct DeviceMemory {
  DeviceMemory(VkDevice, VkPhysicalDevice, VkBuffer buffer,
               VkMemoryPropertyFlags properties);
  ~DeviceMemory();
  DeviceMemory(DeviceMemory &&) noexcept;

  auto operator=(DeviceMemory &&) -> DeviceMemory & = delete;
  DeviceMemory(const DeviceMemory &) = delete;
  auto operator=(const DeviceMemory &) -> DeviceMemory & = delete;

  VkDevice device;
  VkDeviceMemory memory{};
};

struct DescriptorSetLayout {
  explicit DescriptorSetLayout(VkDevice);
  ~DescriptorSetLayout();

  DescriptorSetLayout(DescriptorSetLayout &&) = delete;
  auto operator=(DescriptorSetLayout &&) -> DescriptorSetLayout & = delete;
  DescriptorSetLayout(const DescriptorSetLayout &) = delete;
  auto operator=(const DescriptorSetLayout &) -> DescriptorSetLayout & = delete;

  VkDevice device;
  VkDescriptorSetLayout descriptorSetLayout{};
};

struct DescriptorPool {
  DescriptorPool(VkDevice, const std::vector<VkImage> &swapChainImages);
  ~DescriptorPool();

  DescriptorPool(DescriptorPool &&) = delete;
  auto operator=(DescriptorPool &&) -> DescriptorPool & = delete;
  DescriptorPool(const DescriptorPool &) = delete;
  auto operator=(const DescriptorPool &) -> DescriptorPool & = delete;

  VkDevice device;
  VkDescriptorPool descriptorPool{};
};
} // namespace vulkan_wrappers
} // namespace sbash64::graphics

#endif
