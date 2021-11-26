#include <sbash64/graphics/glfw-wrappers.hpp>
#include <sbash64/graphics/load-object.hpp>
#include <sbash64/graphics/stbi-wrappers.hpp>
#include <sbash64/graphics/vulkan-wrappers.hpp>

#include <vulkan/vulkan_core.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
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

struct VulkanDescriptor {
  vulkan_wrappers::DescriptorPool pool;
  std::vector<VkDescriptorSet> sets;
};

struct VulkanBufferWithMemory {
  vulkan_wrappers::Buffer buffer;
  vulkan_wrappers::DeviceMemory memory;
};

static auto descriptor(
    const vulkan_wrappers::Device &vulkanDevice,
    const vulkan_wrappers::ImageView &vulkanTextureImageView,
    const vulkan_wrappers::Sampler &vulkanTextureSampler,
    const vulkan_wrappers::DescriptorSetLayout &vulkanDescriptorSetLayout,
    const std::vector<VkImage> &swapChainImages,
    const std::vector<VulkanBufferWithMemory> &vulkanUniformBuffers)
    -> VulkanDescriptor {
  vulkan_wrappers::DescriptorPool vulkanDescriptorPool{vulkanDevice.device,
                                                       swapChainImages};
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
    bufferInfo.range = sizeof(UniformBufferObject);

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

static auto present(VkQueue presentQueue,
                    const vulkan_wrappers::Swapchain &swapchain,
                    uint32_t imageIndex, VkSemaphore signalSemaphore)
    -> VkResult {
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

static void submit(const vulkan_wrappers::Device &device, VkQueue graphicsQueue,
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

struct VulkanImage {
  vulkan_wrappers::Image image;
  vulkan_wrappers::ImageView view;
  vulkan_wrappers::DeviceMemory memory;
};

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

static auto bufferWithMemory(VkDevice device, VkPhysicalDevice physicalDevice,
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
  std::vector<float> rotationAnglesDegrees;
  glm::vec3 position;
};

struct FixedPointVector3D {
  int x;
  int y;
  int z;
};

struct GlfwCallback {
  Mouse mouse{};
  Camera camera;
  bool frameBufferResized{};
};

static auto viewMatrix(const Camera &camera) -> glm::mat4 {
  return glm::translate(glm::mat4{1.F}, camera.position) *
         glm::rotate(
             glm::rotate(
                 glm::rotate(glm::mat4{1.F},
                             glm::radians(camera.rotationAnglesDegrees[0]),
                             glm::vec3{1.F, 0.F, 0.F}),
                 glm::radians(camera.rotationAnglesDegrees[1]),
                 glm::vec3{0.F, 1.F, 0.F}),
             glm::radians(camera.rotationAnglesDegrees[2]),
             glm::vec3{0.F, 0.F, 1.F});
}

static void updateCameraAndMouse(GlfwCallback *callback, float x, float y) {
  const auto dy{callback->mouse.position.y - y};
  const auto dx{callback->mouse.position.x - x};
  if (callback->mouse.leftPressed) {
    const auto rotationSpeed{1.F};
    callback->camera.rotationAnglesDegrees[0] += dy * rotationSpeed;
    callback->camera.rotationAnglesDegrees[1] -= dx * rotationSpeed;
  }
  if (callback->mouse.rightPressed) {
    callback->camera.position += glm::vec3(-dx * .01F, -dy * .01F, .0F);
  }
  if (callback->mouse.middlePressed) {
    const auto zoomSpeed{1.F};
    callback->camera.position += glm::vec3(0.F, 0.F, dy * .005F * zoomSpeed);
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
                                const Camera &camera, glm::mat4 perspective,
                                glm::vec3 translation, float scale = 1,
                                float rotationAngleDegrees = 0) {
  UniformBufferObject ubo{};
  perspective[1][1] *= -1;
  ubo.mvp = perspective * viewMatrix(camera) *
            glm::translate(glm::mat4{1.F}, translation) *
            glm::rotate(glm::mat4{1.F}, glm::radians(rotationAngleDegrees),
                        glm::vec3{0.F, 1.F, 0.F}) *
            glm::scale(glm::vec3{scale, scale, scale});
  copy(device.device, memory.memory, &ubo, sizeof(ubo));
}

static auto frameImage(VkDevice device, VkPhysicalDevice physicalDevice,
                       VkSurfaceKHR surface, GLFWwindow *window,
                       VkFormat format, VkImageUsageFlags usageFlags,
                       VkImageAspectFlags aspectFlags) -> VulkanImage {
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

static auto semaphores(VkDevice device, int n)
    -> std::vector<vulkan_wrappers::Semaphore> {
  std::vector<vulkan_wrappers::Semaphore> semaphores;
  generate_n(back_inserter(semaphores), n,
             [device]() { return vulkan_wrappers::Semaphore{device}; });
  return semaphores;
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
  transform(
      playerTextureImages.begin(), playerTextureImages.end(),
      back_inserter(descriptors),
      [&vulkanDevice, &vulkanTextureSampler, &vulkanDescriptorSetLayout,
       &swapChainImages, &vulkanUniformBuffers](const VulkanImage &image) {
        return graphics::descriptor(
            vulkanDevice, image.view, vulkanTextureSampler,
            vulkanDescriptorSetLayout, swapChainImages, vulkanUniformBuffers);
      });
  return descriptors;
}

static void begin(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
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

static void beginWithThrow(VkCommandBuffer commandBuffer) {
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  throwOnError(
      [&]() { return vkBeginCommandBuffer(commandBuffer, &beginInfo); },
      "failed to begin recording command buffer!");
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
  const glfw_wrappers::Window glfwWindow{1280, 720};
  GlfwCallback glfwCallback{};
  glfwSetWindowUserPointer(glfwWindow.window, &glfwCallback);
  glfwSetFramebufferSizeCallback(glfwWindow.window, onFramebufferResize);
  glfwSetCursorPosCallback(glfwWindow.window, onCursorPositionChanged);
  glfwSetMouseButtonCallback(glfwWindow.window, onMouseButton);
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
  const vulkan_wrappers::DescriptorSetLayout vulkanDescriptorSetLayout{
      vulkanDevice.device};

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
  const vulkan_wrappers::Pipeline vulkanPipeline{
      vulkanDevice.device,         vulkanPhysicalDevice,
      vulkanSurface.surface,       vulkanPipelineLayout.pipelineLayout,
      vulkanRenderPass.renderPass, vertexShaderCodePath,
      fragmentShaderCodePath,      glfwWindow.window};

  const auto playerDrawables{
      drawables(vulkanDevice.device, vulkanPhysicalDevice,
                vulkanCommandPool.commandPool, graphicsQueue, playerObjects)};

  const auto worldDrawables{drawables(vulkanDevice.device, vulkanPhysicalDevice,
                                      vulkanCommandPool.commandPool,
                                      graphicsQueue, worldObjects)};

  for (auto i{0U}; i < vulkanCommandBuffers.commandBuffers.size(); i++) {
    beginWithThrow(vulkanCommandBuffers.commandBuffers[i]);
    begin(vulkanPhysicalDevice, vulkanSurface.surface,
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
  auto worldOrigin = glm::vec3{30.F, 0.F, 0.F};
  glfwCallback.camera.rotationAnglesDegrees = {0.F, 0.F, 0.F};
  glfwCallback.camera.position = glm::vec3{0.F, -10.F, -10.F};
  FixedPointVector3D playerDisplacement{-400, 0, 0};
  while (!recreatingSwapChain) {
    if (glfwWindowShouldClose(glfwWindow.window) != 0) {
      break;
    }

    glfwPollEvents();
    {
      constexpr auto playerRunAcceleration{2};
      constexpr auto playerJumpAcceleration{6};
      const RationalNumber gravity{-1, 4};
      if (pressing(glfwWindow.window, GLFW_KEY_Z)) {
        worldOrigin += glm::vec3{
            pressing(glfwWindow.window, GLFW_KEY_LEFT_SHIFT) ? 1.F : -1.F, 0.F,
            0.F};
      }
      if (pressing(glfwWindow.window, GLFW_KEY_X)) {
        worldOrigin += glm::vec3{
            0.F, pressing(glfwWindow.window, GLFW_KEY_LEFT_SHIFT) ? 1.F : -1.F,
            0.F};
      }
      if (pressing(glfwWindow.window, GLFW_KEY_C)) {
        worldOrigin += glm::vec3{
            0.F, 0.F,
            pressing(glfwWindow.window, GLFW_KEY_LEFT_SHIFT) ? 1.F : -1.F};
      }
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
                         0.1F, 100.F);
    const glm::vec3 playerPosition{playerDisplacement.x / 10.F,
                                   playerDisplacement.y / 10.F,
                                   playerDisplacement.z / 10.F};
    updateUniformBuffer(
        vulkanDevice, playerUniformBuffersWithMemory[imageIndex].memory,
        glfwCallback.camera, projection, playerPosition, 1, -90.F);
    updateUniformBuffer(vulkanDevice,
                        worldUniformBuffersWithMemory[imageIndex].memory,
                        glfwCallback.camera, projection, worldOrigin, 20.F);

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
