#ifndef SBASH64_GRAPHICS_VULKAN_WRAPPERS_HPP_
#define SBASH64_GRAPHICS_VULKAN_WRAPPERS_HPP_

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <functional>
#include <vector>

namespace sbash64::graphics {
constexpr std::array<const char *, 1> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

auto supportsGraphics(const VkQueueFamilyProperties &) -> bool;
auto vulkanCountFromPhysicalDevice(
    VkPhysicalDevice, const std::function<void(VkPhysicalDevice, uint32_t *)> &)
    -> uint32_t;
auto queueFamilyPropertiesCount(VkPhysicalDevice) -> uint32_t;
auto queueFamilyProperties(VkPhysicalDevice, uint32_t count)
    -> std::vector<VkQueueFamilyProperties>;
auto graphicsSupportingQueueFamilyIndex(
    const std::vector<VkQueueFamilyProperties> &) -> uint32_t;
auto graphicsSupportingQueueFamilyIndex(VkPhysicalDevice) -> uint32_t;
auto presentSupportingQueueFamilyIndex(VkPhysicalDevice, VkSurfaceKHR)
    -> uint32_t;
auto swapExtent(VkPhysicalDevice, VkSurfaceKHR, GLFWwindow *) -> VkExtent2D;
auto swapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &)
    -> VkSurfaceFormatKHR;
auto findDepthFormat(VkPhysicalDevice) -> VkFormat;

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
  Device(VkPhysicalDevice, VkSurfaceKHR);
  ~Device();

  Device(const Device &) = delete;
  auto operator=(const Device &) -> Device & = delete;
  Device(Device &&) = delete;
  auto operator=(Device &&) -> Device & = delete;

  VkDevice device{};
};

struct Surface {
  Surface(VkInstance, GLFWwindow *);
  ~Surface();

  Surface(const Surface &) = delete;
  auto operator=(const Surface &) -> Surface & = delete;
  Surface(Surface &&) = delete;
  auto operator=(Surface &&) -> Surface & = delete;

  VkInstance instance;
  VkSurfaceKHR surface{};
};

struct Swapchain {
  Swapchain(VkDevice, VkPhysicalDevice, VkSurfaceKHR, GLFWwindow *);
  ~Swapchain();

  Swapchain(const Swapchain &) = delete;
  auto operator=(const Swapchain &) -> Swapchain & = delete;
  Swapchain(Swapchain &&) = delete;
  auto operator=(Swapchain &&) -> Swapchain & = delete;

  VkDevice device;
  VkSwapchainKHR swapChain{};
};

struct Image {
  Image(VkDevice, uint32_t width, uint32_t height, VkFormat, VkImageTiling,
        VkImageUsageFlags, VkMemoryPropertyFlags);
  ~Image();

  Image(const Image &) = delete;
  auto operator=(const Image &) -> Image & = delete;
  Image(Image &&) = delete;
  auto operator=(Image &&) -> Image & = delete;

  VkDevice device;
  VkImage image{};
};

struct ImageView {
  ImageView(VkDevice, VkImage, VkFormat, VkImageAspectFlags);
  ~ImageView();
  ImageView(ImageView &&) noexcept;

  auto operator=(ImageView &&) -> ImageView & = delete;
  ImageView(const ImageView &) = delete;
  auto operator=(const ImageView &) -> ImageView & = delete;

  VkDevice device;
  VkImageView view{};
};

struct Sampler {
  Sampler(VkDevice, VkPhysicalDevice);
  ~Sampler();

  Sampler(Sampler &&) = delete;
  auto operator=(Sampler &&) -> Sampler & = delete;
  Sampler(const Sampler &) = delete;
  auto operator=(const Sampler &) -> Sampler & = delete;

  VkDevice device;
  VkSampler sampler{};
};

struct ShaderModule {
  ShaderModule(VkDevice, const std::vector<char> &code);
  ~ShaderModule();

  ShaderModule(ShaderModule &&) = delete;
  auto operator=(ShaderModule &&) -> ShaderModule & = delete;
  ShaderModule(const ShaderModule &) = delete;
  auto operator=(const ShaderModule &) -> ShaderModule & = delete;

  VkDevice device;
  VkShaderModule module{};
};

struct PipelineLayout {
  PipelineLayout(VkDevice, VkDescriptorSetLayout);
  ~PipelineLayout();

  PipelineLayout(PipelineLayout &&) = delete;
  auto operator=(PipelineLayout &&) -> PipelineLayout & = delete;
  PipelineLayout(const PipelineLayout &) = delete;
  auto operator=(const PipelineLayout &) -> PipelineLayout & = delete;

  VkDevice device;
  VkPipelineLayout pipelineLayout{};
};

struct RenderPass {
  RenderPass(VkDevice, VkPhysicalDevice, VkSurfaceKHR);
  ~RenderPass();

  RenderPass(RenderPass &&) = delete;
  auto operator=(RenderPass &&) -> RenderPass & = delete;
  RenderPass(const RenderPass &) = delete;
  auto operator=(const RenderPass &) -> RenderPass & = delete;

  VkDevice device;
  VkRenderPass renderPass{};
};

struct Pipeline {
  Pipeline(VkDevice, VkPhysicalDevice, VkSurfaceKHR, VkPipelineLayout,
           VkRenderPass, const std::string &vertexShaderCodePath,
           const std::string &fragmentShaderCodePath, GLFWwindow *);
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
              const std::vector<VkImageView> &, GLFWwindow *);
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
  Buffer(VkDevice, VkBufferUsageFlags, VkDeviceSize);
  ~Buffer();
  Buffer(Buffer &&) noexcept;

  auto operator=(Buffer &&) -> Buffer & = delete;
  Buffer(const Buffer &) = delete;
  auto operator=(const Buffer &) -> Buffer & = delete;

  VkDevice device;
  VkBuffer buffer{};
};

struct DeviceMemory {
  DeviceMemory(VkDevice, VkPhysicalDevice, VkMemoryPropertyFlags,
               const VkMemoryRequirements &);
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
