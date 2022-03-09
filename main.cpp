#include <memory>
#include <sbash64/graphics/glfw-wrappers.hpp>
#include <sbash64/graphics/load-object.hpp>
#include <sbash64/graphics/stbi-wrappers.hpp>
#include <sbash64/graphics/vulkan-wrappers.hpp>

#include <vulkan/vulkan_core.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/dual_quaternion.hpp>
#include <glm/gtx/transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace sbash64::graphics {
constexpr auto clamp(int velocity, int limit) -> int {
  return std::clamp(velocity, -limit, limit);
}

constexpr auto withFriction(int velocity, int friction) -> int {
  return (velocity < 0 ? -1 : 1) * std::max(0, std::abs(velocity) - friction);
}

enum class JumpState { grounded, started, released };

struct RationalNumber {
  int numerator;
  int denominator;
};

constexpr auto operator+=(RationalNumber &a, RationalNumber b)
    -> RationalNumber & {
  const auto smallerDenominator{std::min(a.denominator, b.denominator)};
  const auto largerDenominator{std::max(a.denominator, b.denominator)};
  auto commonDenominator{smallerDenominator};
  auto candidateDenominator{largerDenominator};
  while (true) {
    while (commonDenominator <
           std::min(std::numeric_limits<int>::max() - smallerDenominator,
                    candidateDenominator))
      commonDenominator += smallerDenominator;
    if (commonDenominator != candidateDenominator) {
      if (candidateDenominator <
          std::numeric_limits<int>::max() - largerDenominator)
        candidateDenominator += largerDenominator;
      else
        return a;
    } else {
      a.numerator = a.numerator * commonDenominator / a.denominator +
                    b.numerator * commonDenominator / b.denominator;
      a.denominator = commonDenominator;
      return a;
    }
  }
}

constexpr auto operator+=(RationalNumber &a, int b) -> RationalNumber {
  a.numerator += a.denominator * b;
  return a;
}

constexpr auto absoluteValue(int a) -> int { return a < 0 ? -a : a; }

constexpr auto round(RationalNumber a) -> int {
  const auto division{a.numerator / a.denominator};
  if (absoluteValue(a.numerator) % a.denominator <
      (absoluteValue(a.denominator) + 1) / 2)
    return division;
  return ((a.numerator < 0) ^ (a.denominator < 0)) != 0 ? division - 1
                                                        : division + 1;
}

struct UniformBufferObject {
  alignas(16) glm::mat4 mvp;
};

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

static void copy(VkDevice device, VkPhysicalDevice physicalDevice,
                 VkCommandPool commandPool, VkQueue graphicsQueue,
                 VkImage destinationImage,
                 const stbi_wrappers::Image &sourceImage) {
  const auto imageSize{
      static_cast<VkDeviceSize>(sourceImage.width * sourceImage.height * 4)};
  const vulkan_wrappers::Buffer stagingBuffer{
      device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, imageSize};
  const auto memory{bufferMemory(device, physicalDevice, stagingBuffer.buffer,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
  copy(device, memory.memory, sourceImage.pixels, imageSize);
  const auto mipLevels{static_cast<uint32_t>(std::floor(std::log2(
                           std::max(sourceImage.width, sourceImage.height)))) +
                       1};
  transitionImageLayout(device, commandPool, graphicsQueue, destinationImage,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
  copyBufferToImage(device, commandPool, graphicsQueue, stagingBuffer.buffer,
                    destinationImage, static_cast<uint32_t>(sourceImage.width),
                    static_cast<uint32_t>(sourceImage.height));
  generateMipmaps(device, physicalDevice, commandPool, graphicsQueue,
                  destinationImage, VK_FORMAT_R8G8B8A8_SRGB, sourceImage.width,
                  sourceImage.height, mipLevels);
}

static auto textureImage(VkDevice device, VkPhysicalDevice physicalDevice,
                         VkCommandPool commandPool, VkQueue graphicsQueue,
                         const std::string &path) -> VulkanImage {
  const stbi_wrappers::Image stbiImage{path};
  const auto mipLevels{static_cast<uint32_t>(std::floor(std::log2(
                           std::max(stbiImage.width, stbiImage.height)))) +
                       1};
  vulkan_wrappers::Image image{
      device,
      static_cast<uint32_t>(stbiImage.width),
      static_cast<uint32_t>(stbiImage.height),
      VK_FORMAT_R8G8B8A8_SRGB,
      VK_IMAGE_TILING_OPTIMAL,
      static_cast<uint32_t>(VK_IMAGE_USAGE_TRANSFER_SRC_BIT) |
          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      mipLevels,
      VK_SAMPLE_COUNT_1_BIT};
  auto memory{imageMemory(device, physicalDevice, image.image,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
  copy(device, physicalDevice, commandPool, graphicsQueue, image.image,
       stbiImage);
  vulkan_wrappers::ImageView view{device, image.image, VK_FORMAT_R8G8B8A8_SRGB,
                                  VK_IMAGE_ASPECT_COLOR_BIT, mipLevels};
  return VulkanImage{std::move(image), std::move(view), std::move(memory)};
}

struct VulkanDrawable {
  VulkanBufferWithMemory vertexBufferWithMemory;
  VulkanBufferWithMemory indexBufferWithMemory;
};

static auto drawable(VkDevice device, VkPhysicalDevice physicalDevice,
                     VkCommandPool commandPool, VkQueue graphicsQueue,
                     const Object &object) -> VulkanDrawable {
  return VulkanDrawable{
      bufferWithMemory(device, physicalDevice, commandPool, graphicsQueue,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                       object.vertices.data(),
                       sizeof(object.vertices[0]) * object.vertices.size()),
      bufferWithMemory(device, physicalDevice, commandPool, graphicsQueue,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                       object.indices.data(),
                       sizeof(object.indices[0]) * object.indices.size())};
}

struct ScreenPoint {
  float x;
  float y;
};

struct Mouse {
  ScreenPoint position;
  bool leftPressed;
  bool rightPressed;
  bool middlePressed;
};

struct Camera {
  float yaw;
  float pitch;
};

struct FixedPointVector3D {
  int x;
  int y;
  int z;
};

struct GlfwCallback {
  Mouse mouse{};
  Camera camera{};
  bool frameBufferResized{};
};

static void updateCameraAndMouse(GlfwCallback *callback, float x, float y) {
  const auto dy{callback->mouse.position.y - y};
  const auto dx{callback->mouse.position.x - x};
  if (callback->mouse.leftPressed) {
    const auto sensitivity = 0.5F;
    callback->camera.yaw += dx * sensitivity;
    callback->camera.pitch =
        std::clamp(callback->camera.pitch + dy * sensitivity, 1.F, 179.F);
  }
  callback->mouse.position = {x, y};
}

static void onCursorPositionChanged(GLFWwindow *window, double x, double y) {
  updateCameraAndMouse(
      static_cast<GlfwCallback *>(glfwGetWindowUserPointer(window)),
      static_cast<float>(x), static_cast<float>(y));
}

static void onMouseButton(GLFWwindow *window, int button, int action,
                          int /*mods*/) {
  auto *const glfwCallback =
      static_cast<GlfwCallback *>(glfwGetWindowUserPointer(window));
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    if (action == GLFW_PRESS)
      glfwCallback->mouse.leftPressed = true;
    else if (action == GLFW_RELEASE)
      glfwCallback->mouse.leftPressed = false;
  } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    if (action == GLFW_PRESS)
      glfwCallback->mouse.rightPressed = true;
    else if (action == GLFW_RELEASE)
      glfwCallback->mouse.rightPressed = false;
  } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
    if (action == GLFW_PRESS)
      glfwCallback->mouse.middlePressed = true;
    else if (action == GLFW_RELEASE)
      glfwCallback->mouse.middlePressed = false;
  }
}

static void onFramebufferResize(GLFWwindow *window, int /*width*/,
                                int /*height*/) {
  auto *const glfwCallback =
      static_cast<GlfwCallback *>(glfwGetWindowUserPointer(window));
  glfwCallback->frameBufferResized = true;
}

static void updateUniformBuffer(const vulkan_wrappers::Device &device,
                                const vulkan_wrappers::DeviceMemory &memory,
                                glm::mat4 view, glm::mat4 perspective,
                                glm::vec3 translation, float scale = 1,
                                float rotationAngleDegrees = 0) {
  UniformBufferObject ubo{};
  perspective[1][1] *= -1;
  ubo.mvp = perspective * view * glm::translate(glm::mat4{1.F}, translation) *
            glm::rotate(glm::mat4{1.F}, glm::radians(rotationAngleDegrees),
                        glm::vec3{0.F, 1.F, 0.F}) *
            glm::scale(glm::vec3{scale, scale, scale});
  copy(device.device, memory.memory, &ubo, sizeof(ubo));
}

static void draw(const std::vector<Object> &objects,
                 const std::vector<VulkanDrawable> &drawables,
                 const vulkan_wrappers::PipelineLayout &pipelineLayout,
                 const std::vector<VulkanDescriptor> &descriptors,
                 VkCommandBuffer commandBuffer, unsigned int i) {
  for (auto j{0U}; j < drawables.size(); ++j) {
    std::array<VkBuffer, 1> vertexBuffers = {
        drawables.at(j).vertexBufferWithMemory.buffer.buffer};
    std::array<VkDeviceSize, 1> offsets = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers.data(),
                           offsets.data());
    vkCmdBindIndexBuffer(commandBuffer,
                         drawables.at(j).indexBufferWithMemory.buffer.buffer, 0,
                         VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout.pipelineLayout, 0, 1,
                            &descriptors.at(j).sets[i], 0, nullptr);
    vkCmdDrawIndexed(commandBuffer,
                     static_cast<uint32_t>(objects.at(j).indices.size()), 1, 0,
                     0, 0);
  }
}

static auto uniformBufferWithMemory(VkDevice device,
                                    VkPhysicalDevice physicalDevice)
    -> VulkanBufferWithMemory {
  VkDeviceSize bufferSize{sizeof(UniformBufferObject)};
  vulkan_wrappers::Buffer buffer{device, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                 bufferSize};
  auto memory{bufferMemory(device, physicalDevice, buffer.buffer,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
  return VulkanBufferWithMemory{std::move(buffer), std::move(memory)};
}

static auto textureImages(VkDevice device, VkPhysicalDevice physicalDevice,
                          VkCommandPool commandPool, VkQueue graphicsQueue,
                          const std::vector<std::string> &textureImagePaths)
    -> std::vector<VulkanImage> {
  std::vector<VulkanImage> textureImages;
  transform(textureImagePaths.begin(), textureImagePaths.end(),
            back_inserter(textureImages),
            [device, physicalDevice, commandPool,
             graphicsQueue](const std::string &path) {
              return textureImage(device, physicalDevice, commandPool,
                                  graphicsQueue, path);
            });
  return textureImages;
}

static auto drawables(VkDevice device, VkPhysicalDevice physicalDevice,
                      VkCommandPool commandPool, VkQueue graphicsQueue,
                      const std::vector<Object> &objects)
    -> std::vector<VulkanDrawable> {
  std::vector<VulkanDrawable> drawables;
  transform(objects.begin(), objects.end(), back_inserter(drawables),
            [device, physicalDevice, commandPool,
             graphicsQueue](const Object &object) {
              return drawable(device, physicalDevice, commandPool,
                              graphicsQueue, object);
            });
  return drawables;
}

static auto
uniformBuffersWithMemory(VkDevice device, VkPhysicalDevice physicalDevice,
                         const std::vector<VkImage> &swapChainImages)
    -> std::vector<VulkanBufferWithMemory> {
  std::vector<VulkanBufferWithMemory> uniformBuffersWithMemory;
  generate_n(back_inserter(uniformBuffersWithMemory), swapChainImages.size(),
             [device, physicalDevice]() {
               return uniformBufferWithMemory(device, physicalDevice);
             });
  return uniformBuffersWithMemory;
}

static auto descriptors(
    const vulkan_wrappers::Device &vulkanDevice,
    const vulkan_wrappers::Sampler &vulkanTextureSampler,
    const vulkan_wrappers::DescriptorSetLayout &vulkanDescriptorSetLayout,
    const std::vector<VkImage> &swapChainImages,
    const std::vector<VulkanBufferWithMemory> &vulkanUniformBuffers,
    const std::vector<VulkanImage> &playerTextureImages)
    -> std::vector<VulkanDescriptor> {
  std::vector<VulkanDescriptor> descriptors;
  transform(playerTextureImages.begin(), playerTextureImages.end(),
            back_inserter(descriptors),
            [&vulkanDevice, &vulkanTextureSampler, &vulkanDescriptorSetLayout,
             &swapChainImages,
             &vulkanUniformBuffers](const VulkanImage &image) {
              return graphics::descriptor(
                  vulkanDevice, image.view, vulkanTextureSampler,
                  vulkanDescriptorSetLayout, swapChainImages,
                  vulkanUniformBuffers, sizeof(UniformBufferObject));
            });
  return descriptors;
}

static auto textureImagePaths(const std::string &objectPath,
                              const std::vector<Object> &objects)
    -> std::vector<std::string> {
  std::vector<std::string> textureImagePaths;
  transform(objects.begin(), objects.end(), back_inserter(textureImagePaths),
            [&objectPath](const Object &object) {
              return std::filesystem::path{objectPath}.parent_path() /
                     object.textureFileName;
            });
  return textureImagePaths;
}

static auto readTexturedObjects(const std::string &path)
    -> std::vector<Object> {
  auto objects{readObjects(path)};
  erase_if(objects,
           [](const Object &object) { return object.textureFileName.empty(); });
  return objects;
}

static auto pressing(GLFWwindow *window, int key) -> bool {
  return glfwGetKey(window, key) == GLFW_PRESS;
}

static void run(const std::string &vertexShaderCodePath,
                const std::string &fragmentShaderCodePath,
                const std::string &playerObjectPath,
                const std::string &worldObjectPath) {
  const glfw_wrappers::Init glfwInitialization;
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  const glfw_wrappers::Window glfwWindow{1280, 960};
  GlfwCallback glfwCallback{};
  glfwSetWindowUserPointer(glfwWindow.window, &glfwCallback);
  glfwSetCursorPosCallback(glfwWindow.window, onCursorPositionChanged);
  glfwSetMouseButtonCallback(glfwWindow.window, onMouseButton);
  glfwSetFramebufferSizeCallback(glfwWindow.window, onFramebufferResize);
  const vulkan_wrappers::Instance vulkanInstance;
  const vulkan_wrappers::Surface vulkanSurface{vulkanInstance.instance,
                                               glfwWindow.window};
  auto *const vulkanPhysicalDevice{
      suitableDevice(vulkanInstance.instance, vulkanSurface.surface)};
  const vulkan_wrappers::Device vulkanDevice{vulkanPhysicalDevice,
                                             vulkanSurface.surface};
  const vulkan_wrappers::Swapchain vulkanSwapchain{
      vulkanDevice.device, vulkanPhysicalDevice, vulkanSurface.surface,
      glfwWindow.window};
  const auto swapChainImages{graphics::swapChainImages(
      vulkanDevice.device, vulkanSwapchain.swapChain)};
  const auto swapChainImageViews{
      graphics::swapChainImageViews(vulkanDevice.device, vulkanPhysicalDevice,
                                    vulkanSurface.surface, swapChainImages)};
  const auto vulkanColorImage{frameImage(
      vulkanDevice.device, vulkanPhysicalDevice, vulkanSurface.surface,
      glfwWindow.window,
      swapSurfaceFormat(vulkanPhysicalDevice, vulkanSurface.surface).format,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
          VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
      VK_IMAGE_ASPECT_COLOR_BIT)};
  const auto vulkanDepthImage{frameImage(
      vulkanDevice.device, vulkanPhysicalDevice, vulkanSurface.surface,
      glfwWindow.window, findDepthFormat(vulkanPhysicalDevice),
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT)};
  const vulkan_wrappers::RenderPass vulkanRenderPass{
      vulkanDevice.device, vulkanPhysicalDevice, vulkanSurface.surface};
  std::vector<vulkan_wrappers::Framebuffer> vulkanFrameBuffers;
  transform(swapChainImageViews.begin(), swapChainImageViews.end(),
            back_inserter(vulkanFrameBuffers),
            [&vulkanDevice, vulkanPhysicalDevice, &vulkanSurface,
             &vulkanRenderPass, &vulkanColorImage, &vulkanDepthImage,
             &glfwWindow](const vulkan_wrappers::ImageView &imageView) {
              return vulkan_wrappers::Framebuffer{vulkanDevice.device,
                                                  vulkanPhysicalDevice,
                                                  vulkanSurface.surface,
                                                  vulkanRenderPass.renderPass,
                                                  {vulkanColorImage.view.view,
                                                   vulkanDepthImage.view.view,
                                                   imageView.view},
                                                  glfwWindow.window};
            });
  const auto playerUniformBuffersWithMemory{uniformBuffersWithMemory(
      vulkanDevice.device, vulkanPhysicalDevice, swapChainImages)};
  const auto worldUniformBuffersWithMemory{uniformBuffersWithMemory(
      vulkanDevice.device, vulkanPhysicalDevice, swapChainImages)};
  const vulkan_wrappers::Sampler vulkanTextureSampler{vulkanDevice.device,
                                                      vulkanPhysicalDevice, 13};
  VkQueue graphicsQueue{nullptr};
  vkGetDeviceQueue(vulkanDevice.device,
                   graphicsSupportingQueueFamilyIndex(vulkanPhysicalDevice), 0,
                   &graphicsQueue);
  const vulkan_wrappers::CommandPool vulkanCommandPool{vulkanDevice.device,
                                                       vulkanPhysicalDevice};

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
  const vulkan_wrappers::DescriptorSetLayout vulkanDescriptorSetLayout{
      vulkanDevice.device, {uboLayoutBinding, samplerLayoutBinding}};

  const auto playerObjects{readTexturedObjects(playerObjectPath)};
  const auto playerTextureImages{textureImages(
      vulkanDevice.device, vulkanPhysicalDevice, vulkanCommandPool.commandPool,
      graphicsQueue, textureImagePaths(playerObjectPath, playerObjects))};
  const auto playerTextureImageDescriptors{descriptors(
      vulkanDevice, vulkanTextureSampler, vulkanDescriptorSetLayout,
      swapChainImages, playerUniformBuffersWithMemory, playerTextureImages)};

  const auto worldObjects{readTexturedObjects(worldObjectPath)};
  const auto worldTextureImages{textureImages(
      vulkanDevice.device, vulkanPhysicalDevice, vulkanCommandPool.commandPool,
      graphicsQueue, textureImagePaths(worldObjectPath, worldObjects))};
  const auto worldTextureImageDescriptors{descriptors(
      vulkanDevice, vulkanTextureSampler, vulkanDescriptorSetLayout,
      swapChainImages, worldUniformBuffersWithMemory, worldTextureImages)};

  const vulkan_wrappers::CommandBuffers vulkanCommandBuffers{
      vulkanDevice.device, vulkanCommandPool.commandPool,
      vulkanFrameBuffers.size()};
  const vulkan_wrappers::PipelineLayout vulkanPipelineLayout{
      vulkanDevice.device, vulkanDescriptorSetLayout.descriptorSetLayout};

  std::vector<VkVertexInputAttributeDescription> attributeDescriptions(2);
  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(Vertex, position);

  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(Vertex, textureCoordinate);

  std::vector<VkVertexInputBindingDescription> bindingDescription(1);
  bindingDescription[0].binding = 0;
  bindingDescription[0].stride = sizeof(Vertex);
  bindingDescription[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  const vulkan_wrappers::Pipeline vulkanPipeline{
      vulkanDevice.device,         vulkanPhysicalDevice,
      vulkanSurface.surface,       vulkanPipelineLayout.pipelineLayout,
      vulkanRenderPass.renderPass, vertexShaderCodePath,
      fragmentShaderCodePath,      glfwWindow.window,
      attributeDescriptions,       bindingDescription};

  const auto playerDrawables{
      drawables(vulkanDevice.device, vulkanPhysicalDevice,
                vulkanCommandPool.commandPool, graphicsQueue, playerObjects)};

  const auto worldDrawables{drawables(vulkanDevice.device, vulkanPhysicalDevice,
                                      vulkanCommandPool.commandPool,
                                      graphicsQueue, worldObjects)};

  for (auto i{0U}; i < vulkanCommandBuffers.commandBuffers.size(); i++) {
    throwIfFailsToBegin(vulkanCommandBuffers.commandBuffers[i]);
    beginRenderPass(vulkanPhysicalDevice, vulkanSurface.surface,
                    vulkanFrameBuffers.at(i).framebuffer,
                    vulkanCommandBuffers.commandBuffers[i], glfwWindow.window,
                    vulkanRenderPass.renderPass);
    vkCmdBindPipeline(vulkanCommandBuffers.commandBuffers[i],
                      VK_PIPELINE_BIND_POINT_GRAPHICS, vulkanPipeline.pipeline);
    draw(playerObjects, playerDrawables, vulkanPipelineLayout,
         playerTextureImageDescriptors, vulkanCommandBuffers.commandBuffers[i],
         i);
    draw(worldObjects, worldDrawables, vulkanPipelineLayout,
         worldTextureImageDescriptors, vulkanCommandBuffers.commandBuffers[i],
         i);
    vkCmdEndRenderPass(vulkanCommandBuffers.commandBuffers[i]);
    throwOnError(
        [&]() {
          return vkEndCommandBuffer(vulkanCommandBuffers.commandBuffers[i]);
        },
        "failed to record command buffer!");
  }

  VkQueue presentQueue{nullptr};
  vkGetDeviceQueue(vulkanDevice.device,
                   presentSupportingQueueFamilyIndex(vulkanPhysicalDevice,
                                                     vulkanSurface.surface),
                   0, &presentQueue);
  const auto maxFramesInFlight{2};
  std::vector<vulkan_wrappers::Fence> vulkanInFlightFences;
  generate_n(back_inserter(vulkanInFlightFences), maxFramesInFlight,
             [&vulkanDevice]() {
               return vulkan_wrappers::Fence{vulkanDevice.device};
             });
  const auto vulkanImageAvailableSemaphores{
      semaphores(vulkanDevice.device, maxFramesInFlight)};
  const auto vulkanRenderFinishedSemaphores{
      semaphores(vulkanDevice.device, maxFramesInFlight)};
  std::vector<VkFence> imagesInFlight(swapChainImages.size(), VK_NULL_HANDLE);
  const auto swapChainExtent{swapExtent(
      vulkanPhysicalDevice, vulkanSurface.surface, glfwWindow.window)};

  auto recreatingSwapChain{false};
  auto currentFrame{0U};
  FixedPointVector3D playerVelocity{};
  RationalNumber verticalVelocity{0, 1};
  auto jumpState{JumpState::grounded};
  auto worldOrigin = glm::vec3{0.F, 0.F, 0.F};
  glfwCallback.camera.yaw = -90;
  glfwCallback.camera.pitch = 15;
  FixedPointVector3D playerDisplacement{1000, 0, -500};
  while (!recreatingSwapChain) {
    if (glfwWindowShouldClose(glfwWindow.window) != 0) {
      break;
    }

    glfwPollEvents();
    {
      constexpr auto playerRunAcceleration{2};
      constexpr auto playerJumpAcceleration{6};
      const RationalNumber gravity{-1, 4};
      if (pressing(glfwWindow.window, GLFW_KEY_A)) {
        playerVelocity.x += playerRunAcceleration;
      }
      if (pressing(glfwWindow.window, GLFW_KEY_D)) {
        playerVelocity.x -= playerRunAcceleration;
      }
      if (pressing(glfwWindow.window, GLFW_KEY_W)) {
        playerVelocity.z += playerRunAcceleration;
      }
      if (pressing(glfwWindow.window, GLFW_KEY_S)) {
        playerVelocity.z -= playerRunAcceleration;
      }
      if (pressing(glfwWindow.window, GLFW_KEY_SPACE) &&
          jumpState == JumpState::grounded) {
        jumpState = JumpState::started;
        verticalVelocity += playerJumpAcceleration;
      }
      verticalVelocity += gravity;
      if (glfwGetKey(glfwWindow.window, GLFW_KEY_SPACE) != GLFW_PRESS &&
          jumpState == JumpState::started) {
        jumpState = JumpState::released;
        if (verticalVelocity.numerator > 0)
          verticalVelocity = {0, 1};
      }
      constexpr auto playerMaxGroundSpeed{4};
      constexpr auto groundFriction{1};
      playerVelocity.x = withFriction(
          clamp(playerVelocity.x, playerMaxGroundSpeed), groundFriction);
      playerVelocity.z = withFriction(
          clamp(playerVelocity.z, playerMaxGroundSpeed), groundFriction);
      playerDisplacement.x += playerVelocity.x;
      playerDisplacement.z += playerVelocity.z;
      playerDisplacement.y += round(verticalVelocity);
      if (playerDisplacement.y < 0) {
        jumpState = JumpState::grounded;
        playerDisplacement.y = 0;
        verticalVelocity = {0, 1};
      }
    }
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

    const auto projection =
        glm::perspective(glm::radians(45.F),
                         static_cast<float>(swapChainExtent.width) /
                             static_cast<float>(swapChainExtent.height),
                         100.F, 10000.F);
    const glm::vec3 playerPosition{playerDisplacement.x * 3.F,
                                   playerDisplacement.y * 3.F,
                                   playerDisplacement.z * 3.F};
    const auto view = glm::lookAt(
        playerPosition +
            600.F * glm::normalize(glm::vec3{
                        std::cos(glm::radians(glfwCallback.camera.yaw)) *
                            std::cos(glm::radians(glfwCallback.camera.pitch)),
                        std::sin(glm::radians(glfwCallback.camera.pitch)),
                        std::sin(glm::radians(glfwCallback.camera.yaw)) *
                            std::cos(glm::radians(glfwCallback.camera.pitch))}),
        playerPosition, glm::vec3(0, 1, 0));
    updateUniformBuffer(vulkanDevice,
                        playerUniformBuffersWithMemory[imageIndex].memory, view,
                        projection, playerPosition, 20, -90.F);
    updateUniformBuffer(vulkanDevice,
                        worldUniformBuffersWithMemory[imageIndex].memory, view,
                        projection, worldOrigin);

    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
      vkWaitForFences(vulkanDevice.device, 1, &imagesInFlight[imageIndex],
                      VK_TRUE, UINT64_MAX);

    imagesInFlight[imageIndex] = vulkanInFlightFences[currentFrame].fence;
    submit(vulkanDevice, graphicsQueue,
           vulkanImageAvailableSemaphores[currentFrame].semaphore,
           vulkanRenderFinishedSemaphores[currentFrame].semaphore,
           vulkanInFlightFences[currentFrame].fence,
           vulkanCommandBuffers.commandBuffers[imageIndex]);
    {
      const auto result{
          present(presentQueue, vulkanSwapchain, imageIndex,
                  vulkanRenderFinishedSemaphores[currentFrame].semaphore)};
      if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
          glfwCallback.frameBufferResized) {
        glfwCallback.frameBufferResized = false;
        prepareForSwapChainRecreation(vulkanDevice.device, glfwWindow.window);
        recreatingSwapChain = true;
      } else if (result != VK_SUCCESS)
        throw std::runtime_error("failed to present swap chain image!");
    }
    if (++currentFrame == maxFramesInFlight)
      currentFrame = 0;
  }
  vkDeviceWaitIdle(vulkanDevice.device);
}
} // namespace sbash64::graphics

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

struct Primitive {
  uint32_t firstIndex;
  uint32_t indexCount;
  int32_t materialIndex;
};

struct Mesh {
  std::vector<Primitive> primitives;
};

struct Node {
  Node *parent{};
  uint32_t index{};
  std::vector<std::unique_ptr<Node>> children;
  Mesh mesh;
  glm::vec3 translation{};
  glm::vec3 scale{1.0F};
  glm::quat rotation{};
  int32_t skin = -1;
  glm::mat4 matrix{};
};

static auto getLocalMatrix(const Node &node) -> glm::mat4 {
  return glm::translate(glm::mat4(1.0F), node.translation) *
         glm::mat4(node.rotation) * glm::scale(glm::mat4(1.0F), node.scale) *
         node.matrix;
}

struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec2 uv;
  glm::vec3 color;
  glm::vec4 jointIndices;
  glm::vec4 jointWeights;
};

struct Material {
  glm::vec4 baseColorFactor = glm::vec4(1.0F);
  uint32_t baseColorTextureIndex{};
};

struct Skin {
  std::string name;
  Node *skeletonRoot = nullptr;
  std::vector<glm::mat4> inverseBindMatrices;
  std::vector<Node *> joints;
};

struct AnimationSampler {
  std::string interpolation;
  std::vector<float> inputs;
  std::vector<glm::vec4> outputsVec4;
};

struct AnimationChannel {
  std::string path;
  Node *node{};
  uint32_t samplerIndex{};
};

struct Animation {
  std::string name;
  std::vector<AnimationSampler> samplers;
  std::vector<AnimationChannel> channels;
  float start = std::numeric_limits<float>::max();
  float end = std::numeric_limits<float>::min();
  float currentTime = 0.0F;
};

template <typename T>
static auto span(const tinygltf::Model &model, int index)
    -> std::span<const T> {
  const auto &accessor = model.accessors[index];
  const auto &bufferView = model.bufferViews[accessor.bufferView];
  const auto &buffer = model.buffers[bufferView.buffer];
  return std::span<const T>{
      reinterpret_cast<const T *>(
          &buffer.data[accessor.byteOffset + bufferView.byteOffset]),
      accessor.count};
}

template <typename T>
void fill(std::vector<uint32_t> &indexBuffer, uint32_t vertexStart,
          const tinygltf::Model &model, const tinygltf::Primitive &primitive) {
  for (const auto index : span<T>(model, primitive.indices))
    indexBuffer.push_back(index + vertexStart);
}

static void loadNode(std::vector<std::unique_ptr<Node>> &nodes,
                     const tinygltf::Node &gltfNode,
                     const tinygltf::Model &gltfModel, Node *parent,
                     uint32_t nodeIndex, std::vector<uint32_t> &indexBuffer,
                     std::vector<Vertex> &vertexBuffer) {
  auto node{std::make_unique<Node>()};
  node->parent = parent;
  node->matrix = glm::mat4(1.0F);
  node->index = nodeIndex;
  node->skin = gltfNode.skin;

  // Get the local node matrix
  // It's either made up from translation, rotation, scale or a 4x4 matrix
  if (gltfNode.translation.size() == 3)
    node->translation = glm::make_vec3(gltfNode.translation.data());
  if (gltfNode.rotation.size() == 4) {
    glm::quat q = glm::make_quat(gltfNode.rotation.data());
    node->rotation = glm::mat4(q);
  }
  if (gltfNode.scale.size() == 3)
    node->scale = glm::make_vec3(gltfNode.scale.data());
  if (gltfNode.matrix.size() == 16)
    node->matrix = glm::make_mat4x4(gltfNode.matrix.data());

  for (int i : gltfNode.children)
    loadNode(nodes, gltfModel.nodes[i], gltfModel, node.get(), i, indexBuffer,
             vertexBuffer);

  // If the node contains mesh data, we load vertices and indices from the
  // buffers In glTF this is done via accessors and buffer views
  if (gltfNode.mesh > -1) {
    for (const auto &gltfPrimitive :
         gltfModel.meshes[gltfNode.mesh].primitives) {
      const auto firstIndex = static_cast<uint32_t>(indexBuffer.size());
      const auto vertexStart = static_cast<uint32_t>(vertexBuffer.size());
      uint32_t indexCount = 0;
      bool hasSkin = false;
      // Vertices
      {
        const float *positionBuffer = nullptr;
        const float *normalsBuffer = nullptr;
        const float *texCoordsBuffer = nullptr;
        const uint16_t *jointIndicesBuffer = nullptr;
        const float *jointWeightsBuffer = nullptr;
        size_t vertexCount = 0;

        // Get buffer data for vertex normals
        if (gltfPrimitive.attributes.contains("POSITION")) {
          const auto span{::span<float>(
              gltfModel, gltfPrimitive.attributes.at("POSITION"))};
          positionBuffer = span.data();
          vertexCount = span.size();
        }
        // Get buffer data for vertex normals
        if (gltfPrimitive.attributes.contains("NORMAL")) {
          const auto span{
              ::span<float>(gltfModel, gltfPrimitive.attributes.at("NORMAL"))};
          normalsBuffer = span.data();
        }
        // Get buffer data for vertex texture coordinates
        // glTF supports multiple sets, we only load the first one
        if (gltfPrimitive.attributes.contains("TEXCOORD_0")) {
          const auto span{::span<float>(
              gltfModel, gltfPrimitive.attributes.at("TEXCOORD_0"))};
          texCoordsBuffer = span.data();
        }
        // POI: Get buffer data required for vertex skinning
        // Get vertex joint indices
        if (gltfPrimitive.attributes.contains("JOINTS_0")) {
          const auto span{::span<uint16_t>(
              gltfModel, gltfPrimitive.attributes.at("JOINTS_0"))};
          jointIndicesBuffer = span.data();
        }
        // Get vertex joint weights
        if (gltfPrimitive.attributes.contains("WEIGHTS_0")) {
          const auto span{::span<float>(
              gltfModel, gltfPrimitive.attributes.at("WEIGHTS_0"))};
          jointWeightsBuffer = span.data();
        }

        hasSkin = ((jointIndicesBuffer != nullptr) &&
                   (jointWeightsBuffer != nullptr));

        // Append data to model's vertex buffer
        for (size_t v = 0; v < vertexCount; v++) {
          Vertex vertex{};
          vertex.pos = glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0F);
          vertex.normal = glm::normalize(glm::vec3(
              normalsBuffer != nullptr ? glm::make_vec3(&normalsBuffer[v * 3])
                                       : glm::vec3(0.0F)));
          vertex.uv = texCoordsBuffer != nullptr
                          ? glm::make_vec2(&texCoordsBuffer[v * 2])
                          : glm::vec3(0.0F);
          vertex.color = glm::vec3(1.0F);
          vertex.jointIndices =
              hasSkin ? glm::vec4(glm::make_vec4(&jointIndicesBuffer[v * 4]))
                      : glm::vec4(0.0F);
          vertex.jointWeights = hasSkin
                                    ? glm::make_vec4(&jointWeightsBuffer[v * 4])
                                    : glm::vec4(0.0F);
          vertexBuffer.push_back(vertex);
        }
      }
      // Indices
      {
        const auto &accessor = gltfModel.accessors[gltfPrimitive.indices];
        indexCount += static_cast<uint32_t>(accessor.count);
        switch (accessor.componentType) {
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
          fill<uint32_t>(indexBuffer, vertexStart, gltfModel, gltfPrimitive);
          break;
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
          fill<uint16_t>(indexBuffer, vertexStart, gltfModel, gltfPrimitive);
          break;
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
          fill<uint8_t>(indexBuffer, vertexStart, gltfModel, gltfPrimitive);
          break;
        default:
          std::cerr << "Index component type " << accessor.componentType
                    << " not supported!" << std::endl;
          return;
        }
      }
      Primitive primitive{};
      primitive.firstIndex = firstIndex;
      primitive.indexCount = indexCount;
      primitive.materialIndex = gltfPrimitive.material;
      node->mesh.primitives.push_back(primitive);
    }
  }

  if (parent != nullptr) {
    parent->children.push_back(std::move(node));
  } else {
    nodes.push_back(std::move(node));
  }
}

static auto findNode(Node *parent, uint32_t index) -> Node * {
  if (parent->index == index)
    return parent;
  for (auto &child : parent->children) {
    Node *node = findNode(child.get(), index);
    if (node != nullptr) {
      return node;
    }
  }
  return nullptr;
}

static auto nodeFromIndex(const std::vector<std::unique_ptr<Node>> &nodes,
                          uint32_t index) -> Node * {
  for (const auto &node : nodes) {
    auto *maybeFound = findNode(node.get(), index);
    if (maybeFound != nullptr)
      return maybeFound;
  }
  return nullptr;
}

static auto getNodeMatrix(Node *node) -> glm::mat4 {
  auto nodeMatrix = getLocalMatrix(*node);
  auto *currentParent = node->parent;
  while (currentParent != nullptr) {
    nodeMatrix = getLocalMatrix(*currentParent) * nodeMatrix;
    currentParent = currentParent->parent;
  }
  return nodeMatrix;
}

static auto joints(Node *) -> std::vector<Node *> {
  throw std::runtime_error{"not implemented"};
}

static auto hasSkin(Node *) -> bool {
  throw std::runtime_error{"not implemented"};
}

static auto inverseBindMatrices(Node *) -> std::vector<glm::mat4> {
  throw std::runtime_error{"not implemented"};
}

static auto deviceMemory(Node *) -> VkDeviceMemory {
  throw std::runtime_error{"not implemented"};
}

static void updateJoints(Node *node, VkDevice device) {
  if (hasSkin(node)) {
    glm::mat4 inverseTransform = glm::inverse(getNodeMatrix(node));
    auto joints{::joints(node)};
    size_t numJoints = joints.size();
    auto inverseBindMatrices{::inverseBindMatrices(node)};
    std::vector<glm::mat4> jointMatrices(numJoints);
    for (size_t i = 0; i < numJoints; i++) {
      jointMatrices[i] = getNodeMatrix(joints[i]) * inverseBindMatrices[i];
      jointMatrices[i] = inverseTransform * jointMatrices[i];
    }
    // Update ssbo
    sbash64::graphics::copy(device, deviceMemory(node), jointMatrices.data(),
                            jointMatrices.size() * sizeof(glm::mat4));
  }
  for (const auto &child : node->children)
    updateJoints(child.get(), device);
}

static auto
vulkanImage(void *buffer, VkDeviceSize bufferSize, VkFormat format,
            uint32_t width, uint32_t height, VkDevice device,
            VkPhysicalDevice physicalDevice, VkCommandPool commandPool,
            VkQueue copyQueue,
            VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT)
    -> sbash64::graphics::VulkanImage {
  const sbash64::graphics::vulkan_wrappers::Buffer stagingBuffer{
      device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, bufferSize};

  const auto stagingMemory{sbash64::graphics::bufferMemory(
      device, physicalDevice, stagingBuffer.buffer,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};

  sbash64::graphics::copy(device, stagingMemory.memory, buffer, bufferSize);

  const uint32_t mipLevels = 1;
  sbash64::graphics::vulkan_wrappers::Image image{
      device,
      width,
      height,
      format,
      VK_IMAGE_TILING_OPTIMAL,
      imageUsageFlags | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      mipLevels,
      VK_SAMPLE_COUNT_1_BIT};

  sbash64::graphics::copyBufferToImage(device, commandPool, copyQueue,
                                       stagingBuffer.buffer, image.image, width,
                                       height);

  auto deviceMemory{
      sbash64::graphics::imageMemory(device, physicalDevice, image.image,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};

  sbash64::graphics::vulkan_wrappers::ImageView imageView{
      device, image.image, format, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels};
  return sbash64::graphics::VulkanImage{std::move(image), std::move(imageView),
                                        std::move(deviceMemory)};
}

static void loadGltf(VkDevice device, VkPhysicalDevice physicalDevice,
                     VkQueue copyQueue, VkCommandPool commandPool) {
  tinygltf::TinyGLTF gltf;
  tinygltf::Model gltfModel;
  {
    std::string error;
    std::string warning;
    gltf.LoadASCIIFromFile(&gltfModel, &error, &warning,
                           "/home/seth/Documents/CesiumMan.gltf");
    std::cout << error << '\n';
    std::cout << warning << '\n';
  }

  std::cout << gltfModel.images.size() << '\n';

  // Images can be stored inside the glTF (which is the case for the sample
  // model), so instead of directly loading them from disk, we fetch them
  // from the glTF loader and upload the buffers
  std::vector<sbash64::graphics::VulkanImage> vulkanImages;
  transform(gltfModel.images.begin(), gltfModel.images.end(),
            back_inserter(vulkanImages),
            [device, physicalDevice, commandPool,
             copyQueue](const tinygltf::Image &image) {
              // Get the image data from the glTF loader
              // We convert RGB-only images to RGBA, as most devices don't
              // support RGB-formats in Vulkan
              std::vector<unsigned char> buffer;
              if (image.component == 3) {
                buffer.resize(image.width * image.height * 4);
                unsigned char *rgba = buffer.data();
                const auto *rgb = &image.image[0];
                for (size_t i = 0; i < image.width * image.height; ++i) {
                  memcpy(rgba, rgb, sizeof(unsigned char) * 3);
                  rgba += 4;
                  rgb += 3;
                }
              } else {
                std::span<const unsigned char> gltfBuffer{&image.image[0],
                                                          image.image.size()};
                buffer.resize(gltfBuffer.size());
                copy(gltfBuffer.begin(), gltfBuffer.end(), buffer.begin());
              }
              return vulkanImage(buffer.data(), buffer.size(),
                                 VK_FORMAT_R8G8B8A8_UNORM, image.width,
                                 image.height, device, physicalDevice,
                                 commandPool, copyQueue);
            });

  std::vector<Material> materials;
  transform(
      gltfModel.materials.begin(), gltfModel.materials.end(),
      back_inserter(materials), [](const tinygltf::Material &gltfMaterial) {
        Material material;
        if (gltfMaterial.values.contains("baseColorFactor"))
          material.baseColorFactor = glm::make_vec4(
              gltfMaterial.values.at("baseColorFactor").ColorFactor().data());
        if (gltfMaterial.values.contains("baseColorTexture"))
          material.baseColorTextureIndex =
              gltfMaterial.values.at("baseColorTexture").TextureIndex();
        return material;
      });

  std::vector<int> textureIndices;
  transform(gltfModel.textures.begin(), gltfModel.textures.end(),
            back_inserter(textureIndices),
            [](const tinygltf::Texture &texture) { return texture.source; });

  std::vector<uint32_t> indexBuffer;
  std::vector<Vertex> vertexBuffer;
  std::vector<std::unique_ptr<Node>> nodes;
  for (int i : gltfModel.scenes[0].nodes)
    loadNode(nodes, gltfModel.nodes[i], gltfModel, nullptr, i, indexBuffer,
             vertexBuffer);

  std::vector<Skin> skins;
  std::vector<sbash64::graphics::VulkanBufferWithMemory>
      jointsVulkanBuffersWithMemory;
  transform(
      gltfModel.skins.begin(), gltfModel.skins.end(), back_inserter(skins),
      [&gltfModel, &nodes, &device, &physicalDevice,
       &jointsVulkanBuffersWithMemory](const tinygltf::Skin &gltfSkin) {
        Skin skin;
        skin.name = gltfSkin.name;
        skin.skeletonRoot = nodeFromIndex(nodes, gltfSkin.skeleton);

        for (const auto jointIndex : gltfSkin.joints) {
          auto *node = nodeFromIndex(nodes, jointIndex);
          if (node != nullptr)
            skin.joints.push_back(node);
        }

        if (gltfSkin.inverseBindMatrices > -1) {
          const auto span{
              ::span<glm::mat4>(gltfModel, gltfSkin.inverseBindMatrices)};
          std::vector<glm::mat4> inverseBindMatrices{span.begin(), span.end()};

          // Store inverse bind matrices for this skin in a shader storage
          // buffer object To keep this sample simple, we create a host visible
          // shader storage buffer

          const VkDeviceSize bufferSize{sizeof(glm::mat4) * span.size()};
          sbash64::graphics::vulkan_wrappers::Buffer jointsVulkanBuffer{
              device, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, bufferSize};
          auto memory{sbash64::graphics::bufferMemory(
              device, physicalDevice, jointsVulkanBuffer.buffer,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
          sbash64::graphics::copy(device, memory.memory,
                                  inverseBindMatrices.data(), bufferSize);
          // VK_CHECK_RESULT(skins[i].ssbo.map());
          jointsVulkanBuffersWithMemory.push_back(
              sbash64::graphics::VulkanBufferWithMemory{
                  std::move(jointsVulkanBuffer), std::move(memory)});
        }
        return skin;
      });
  std::vector<Animation> animations;
  transform(
      gltfModel.animations.begin(), gltfModel.animations.end(),
      back_inserter(animations),
      [&gltfModel, &nodes](const tinygltf::Animation &gltfAnimation) {
        Animation animation;
        animation.name = gltfAnimation.name;
        transform(gltfAnimation.samplers.begin(), gltfAnimation.samplers.end(),
                  back_inserter(animation.samplers),
                  [&gltfModel,
                   &animation](const tinygltf::AnimationSampler &gltfSampler) {
                    AnimationSampler animationSampler;
                    animationSampler.interpolation = gltfSampler.interpolation;
                    for (const auto input :
                         span<float>(gltfModel, gltfSampler.input))
                      animationSampler.inputs.push_back(input);
                    // Adjust animation's start and end times
                    for (const auto input : animationSampler.inputs) {
                      if (input < animation.start)
                        animation.start = input;
                      if (input > animation.end)
                        animation.end = input;
                    }

                    // Read sampler keyframe output translate/rotate/scale
                    // values
                    switch (gltfModel.accessors[gltfSampler.output].type) {
                    case TINYGLTF_TYPE_VEC3:
                      for (const auto v :
                           span<glm::vec3>(gltfModel, gltfSampler.output))
                        animationSampler.outputsVec4.emplace_back(v, 0.0F);
                      break;
                    case TINYGLTF_TYPE_VEC4:
                      for (const auto v :
                           span<glm::vec4>(gltfModel, gltfSampler.output))
                        animationSampler.outputsVec4.push_back(v);
                      break;
                    default:
                      std::cout << "unknown type" << std::endl;
                      break;
                    }
                    return animationSampler;
                  });
        transform(
            gltfAnimation.channels.begin(), gltfAnimation.channels.end(),
            back_inserter(animation.channels),
            [&nodes](const tinygltf::AnimationChannel &gltfAnimationChannel) {
              AnimationChannel animationChannel;
              animationChannel.path = gltfAnimationChannel.target_path;
              animationChannel.samplerIndex = gltfAnimationChannel.sampler;
              animationChannel.node =
                  nodeFromIndex(nodes, gltfAnimationChannel.target_node);
              return animationChannel;
            });
        return animation;
      });
  // Calculate initial pose
  for (const auto &node : nodes)
    updateJoints(node.get(), device);

  // Create and upload vertex and index buffer
  const auto vertexBufferSize = vertexBuffer.size() * sizeof(Vertex);
  const auto indexBufferSize = indexBuffer.size() * sizeof(uint32_t);

  const auto vertexBufferWithMemory{sbash64::graphics::bufferWithMemory(
      device, physicalDevice, commandPool, copyQueue,
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      vertexBuffer.data(), vertexBufferSize)};
  const auto indexBufferWithMemory{sbash64::graphics::bufferWithMemory(
      device, physicalDevice, commandPool, copyQueue,
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      indexBuffer.data(), indexBufferSize)};
}

int main(int argc, char *argv[]) {
  const std::span<char *> arguments{
      argv, static_cast<std::span<char *>::size_type>(argc)};
  if (arguments.size() < 5)
    return EXIT_FAILURE;
  try {
    sbash64::graphics::run(arguments[1], arguments[2], arguments[3],
                           arguments[4]);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
