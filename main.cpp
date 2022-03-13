#include <sbash64/graphics/glfw-wrappers.hpp>
#include <sbash64/graphics/load-object.hpp>
#include <sbash64/graphics/load-scene.hpp>
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
#include <numeric>
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

static auto descriptorSet(const Skin &skin) -> VkDescriptorSet {
  throw std::runtime_error{"unimplemented"};
}

static auto descriptorSet(const Image &image) -> VkDescriptorSet {
  throw std::runtime_error{"unimplemented"};
}

static void drawNode(VkCommandBuffer commandBuffer,
                     VkPipelineLayout pipelineLayout, const Node &node,
                     const Scene &scene) {
  if (!node.mesh.primitives.empty()) {
    // Pass the node's matrix via push constants
    // Traverse the node hierarchy to the top-most parent to get the final
    // matrix of the current node
    auto nodeMatrix = node.matrix;
    const auto *currentParent = node.parent;
    while (currentParent != nullptr) {
      nodeMatrix = currentParent->matrix * nodeMatrix;
      currentParent = currentParent->parent;
    }
    // Pass the final matrix to the vertex shader using push constants
    vkCmdPushConstants(commandBuffer, pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4),
                       &nodeMatrix);
    // Bind SSBO with skin data for this node to set 1
    std::array<VkDescriptorSet, 1> skinDescriptorSets{
        descriptorSet(scene.skins[node.skin])};
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 1, skinDescriptorSets.size(),
                            skinDescriptorSets.data(), 0, nullptr);
    for (const auto &primitive : node.mesh.primitives) {
      if (primitive.indexCount > 0) {
        // Get the texture index for this primitive
        const auto index =
            scene.textureIndices[scene.materials[primitive.materialIndex]
                                     .baseColorTextureIndex];
        std::array<VkDescriptorSet, 1> imageDescriptorSets{
            descriptorSet(scene.images[index])};
        // Bind the descriptor for the current primitive's texture to set 2
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 2, imageDescriptorSets.size(),
                                imageDescriptorSets.data(), 0, nullptr);
        vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1,
                         primitive.firstIndex, 0, 0);
      }
    }
  }
  for (const auto &child : node.children) {
    drawNode(commandBuffer, pipelineLayout, *child, scene);
  }
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

  const vulkan_wrappers::CommandBuffers vulkanDrawCommandBuffers{
      vulkanDevice.device, vulkanCommandPool.commandPool,
      vulkanFrameBuffers.size()};
  const vulkan_wrappers::PipelineLayout vulkanPipelineLayout{
      vulkanDevice.device, {vulkanDescriptorSetLayout.descriptorSetLayout}};

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

  for (auto i{0U}; i < vulkanDrawCommandBuffers.commandBuffers.size(); i++) {
    throwIfFailsToBegin(vulkanDrawCommandBuffers.commandBuffers[i]);
    beginRenderPass(vulkanPhysicalDevice, vulkanSurface.surface,
                    vulkanFrameBuffers.at(i).framebuffer,
                    vulkanDrawCommandBuffers.commandBuffers[i],
                    glfwWindow.window, vulkanRenderPass.renderPass);
    vkCmdBindPipeline(vulkanDrawCommandBuffers.commandBuffers[i],
                      VK_PIPELINE_BIND_POINT_GRAPHICS, vulkanPipeline.pipeline);
    draw(playerObjects, playerDrawables, vulkanPipelineLayout,
         playerTextureImageDescriptors,
         vulkanDrawCommandBuffers.commandBuffers[i], i);
    draw(worldObjects, worldDrawables, vulkanPipelineLayout,
         worldTextureImageDescriptors,
         vulkanDrawCommandBuffers.commandBuffers[i], i);
    // Bind scene matrices descriptor to set 0
    // vkCmdBindDescriptorSets(vulkanDrawCommandBuffers.commandBuffers[i],
    //                         VK_PIPELINE_BIND_POINT_GRAPHICS,
    //                         animatingPipelineLayout, 0, 1, &descriptorSet, 0,
    //                         nullptr);
    // vkCmdBindPipeline(vulkanDrawCommandBuffers.commandBuffers[i],
    //                   animatingPipeline);
    // // All vertices and indices are stored in single buffers, so we only need
    // to
    // // bind once
    // {
    //   VkDeviceSize offsets[1] = {0};
    //   vkCmdBindVertexBuffers(vulkanDrawCommandBuffers.commandBuffers[i], 0,
    //   1,
    //                          &vertices.buffer, offsets);
    //   vkCmdBindIndexBuffer(vulkanDrawCommandBuffers.commandBuffers[i],
    //                        indices.buffer, 0, VK_INDEX_TYPE_UINT32);
    //   // Render all nodes at top-level
    //   for (auto &node : nodes) {
    //     drawNode(commandBuffer, pipelineLayout, *node);
    //   }
    // }
    vkCmdEndRenderPass(vulkanDrawCommandBuffers.commandBuffers[i]);
    throwOnError(
        [&]() {
          return vkEndCommandBuffer(vulkanDrawCommandBuffers.commandBuffers[i]);
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
           vulkanDrawCommandBuffers.commandBuffers[imageIndex]);
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

static auto getLocalMatrix(const Node &node) -> glm::mat4 {
  return glm::translate(glm::mat4(1.0F), node.translation) *
         glm::mat4(node.rotation) * glm::scale(glm::mat4(1.0F), node.scale) *
         node.matrix;
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
    auto joints{graphics::joints(node)};
    size_t numJoints = joints.size();
    auto inverseBindMatrices{graphics::inverseBindMatrices(node)};
    std::vector<glm::mat4> jointMatrices(numJoints);
    for (size_t i = 0; i < numJoints; i++) {
      jointMatrices[i] = getNodeMatrix(joints[i]) * inverseBindMatrices[i];
      jointMatrices[i] = inverseTransform * jointMatrices[i];
    }
    // Update ssbo
    copy(device, deviceMemory(node), jointMatrices.data(),
         jointMatrices.size() * sizeof(glm::mat4));
  }
  for (const auto &child : node->children)
    updateJoints(child.get(), device);
}

static auto
vulkanImage(const void *buffer, VkDeviceSize bufferSize, VkFormat format,
            uint32_t width, uint32_t height, VkDevice device,
            VkPhysicalDevice physicalDevice, VkCommandPool commandPool,
            VkQueue copyQueue,
            VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT)
    -> VulkanImage {
  const vulkan_wrappers::Buffer stagingBuffer{
      device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, bufferSize};

  const auto stagingMemory{
      bufferMemory(device, physicalDevice, stagingBuffer.buffer,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};

  copy(device, stagingMemory.memory, buffer, bufferSize);

  const uint32_t mipLevels = 1;
  vulkan_wrappers::Image image{device,
                               width,
                               height,
                               format,
                               VK_IMAGE_TILING_OPTIMAL,
                               imageUsageFlags |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                               mipLevels,
                               VK_SAMPLE_COUNT_1_BIT};

  copyBufferToImage(device, commandPool, copyQueue, stagingBuffer.buffer,
                    image.image, width, height);

  auto deviceMemory{imageMemory(device, physicalDevice, image.image,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};

  vulkan_wrappers::ImageView imageView{device, image.image, format,
                                       VK_IMAGE_ASPECT_COLOR_BIT, mipLevels};
  return VulkanImage{std::move(image), std::move(imageView),
                     std::move(deviceMemory)};
}

static void updateAnimation(Animation &animation,
                            const std::vector<std::unique_ptr<Node>> &nodes,
                            VkDevice device, float deltaTime) {
  animation.currentTime += deltaTime;
  if (animation.currentTime > animation.end) {
    animation.currentTime -= animation.end;
  }

  for (auto &channel : animation.channels) {
    AnimationSampler &sampler = animation.samplers[channel.samplerIndex];
    for (size_t i = 0; i < sampler.inputs.size() - 1; i++) {
      if (sampler.interpolation != "LINEAR") {
        std::cout << "This sample only supports linear interpolations\n";
        continue;
      }

      // Get the input keyframe values for the current time stamp
      if ((animation.currentTime >= sampler.inputs[i]) &&
          (animation.currentTime <= sampler.inputs[i + 1])) {
        float a = (animation.currentTime - sampler.inputs[i]) /
                  (sampler.inputs[i + 1] - sampler.inputs[i]);
        if (channel.path == "translation") {
          channel.node->translation =
              glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], a);
        }
        if (channel.path == "rotation") {
          glm::quat q1;
          q1.x = sampler.outputsVec4[i].x;
          q1.y = sampler.outputsVec4[i].y;
          q1.z = sampler.outputsVec4[i].z;
          q1.w = sampler.outputsVec4[i].w;

          glm::quat q2;
          q2.x = sampler.outputsVec4[i + 1].x;
          q2.y = sampler.outputsVec4[i + 1].y;
          q2.z = sampler.outputsVec4[i + 1].z;
          q2.w = sampler.outputsVec4[i + 1].w;

          channel.node->rotation = glm::normalize(glm::slerp(q1, q2, a));
        }
        if (channel.path == "scale") {
          channel.node->scale =
              glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], a);
        }
      }
    }
  }
  for (const auto &node : nodes)
    updateJoints(node.get(), device);
}

struct AnimatingUniformBufferObject {
  alignas(16) glm::mat4 projection;
  alignas(16) glm::mat4 view;
};

static auto
animatingPipeline(VkDevice device, VkPhysicalDevice physicalDevice,
                  VkSurfaceKHR surface, VkRenderPass renderPass,
                  GLFWwindow *window, const std::string &vertexShaderCodePath,
                  const std::string &fragmentShaderCodePath,
                  const vulkan_wrappers::PipelineLayout &vulkanPipelineLayout)
    -> vulkan_wrappers::Pipeline {
  std::vector<VkVertexInputAttributeDescription> attributeDescriptions(4);
  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(AnimatingVertex, pos);

  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(AnimatingVertex, uv);

  attributeDescriptions[2].binding = 0;
  attributeDescriptions[2].location = 2;
  attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  attributeDescriptions[2].offset = offsetof(AnimatingVertex, jointIndices);

  attributeDescriptions[3].binding = 0;
  attributeDescriptions[3].location = 3;
  attributeDescriptions[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  attributeDescriptions[3].offset = offsetof(AnimatingVertex, jointWeights);

  std::vector<VkVertexInputBindingDescription> bindingDescription(1);
  bindingDescription[0].binding = 0;
  bindingDescription[0].stride = sizeof(AnimatingVertex);
  bindingDescription[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  return {device,
          physicalDevice,
          surface,
          vulkanPipelineLayout.pipelineLayout,
          renderPass,
          vertexShaderCodePath,
          fragmentShaderCodePath,
          window,
          attributeDescriptions,
          bindingDescription};
}

static auto animatingJointMatricesDescriptorSetLayout(VkDevice device)
    -> vulkan_wrappers::DescriptorSetLayout {
  VkDescriptorSetLayoutBinding jointMatricesLayoutBinding{};
  jointMatricesLayoutBinding.binding = 0;
  jointMatricesLayoutBinding.descriptorCount = 1;
  jointMatricesLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  jointMatricesLayoutBinding.pImmutableSamplers = nullptr;
  jointMatricesLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  return {device, {jointMatricesLayoutBinding}};
}

static auto animatingSamplerDescriptorSetLayout(VkDevice device)
    -> vulkan_wrappers::DescriptorSetLayout {
  VkDescriptorSetLayoutBinding samplerLayoutBinding{};
  samplerLayoutBinding.binding = 0;
  samplerLayoutBinding.descriptorCount = 1;
  samplerLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  samplerLayoutBinding.pImmutableSamplers = nullptr;
  samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  return {device, {samplerLayoutBinding}};
}

static auto animatingUniformBufferObjectDescriptorSetLayout(VkDevice device)
    -> vulkan_wrappers::DescriptorSetLayout {
  VkDescriptorSetLayoutBinding layoutBinding{};
  layoutBinding.binding = 0;
  layoutBinding.descriptorCount = 1;
  layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  layoutBinding.pImmutableSamplers = nullptr;
  layoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  return {device, {layoutBinding}};
}

static auto animatingUniformBufferObjectDescriptorSet(
    VkDevice device, const VulkanBufferWithMemory &bufferWithMemory,
    const vulkan_wrappers::DescriptorPool &descriptorPool,
    const vulkan_wrappers::DescriptorSetLayout &setLayout) -> VkDescriptorSet {
  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
  descriptorSetAllocateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.descriptorPool = descriptorPool.descriptorPool;
  descriptorSetAllocateInfo.pSetLayouts = &setLayout.descriptorSetLayout;
  descriptorSetAllocateInfo.descriptorSetCount = 1;

  VkDescriptorSet descriptorSet = nullptr;
  vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet);

  VkWriteDescriptorSet writeDescriptorSet{};
  writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writeDescriptorSet.dstSet = descriptorSet;
  writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  writeDescriptorSet.dstBinding = 0;

  VkDescriptorBufferInfo descriptorBufferInfo;
  descriptorBufferInfo.buffer = bufferWithMemory.buffer.buffer;
  descriptorBufferInfo.offset = 0;
  descriptorBufferInfo.range = VK_WHOLE_SIZE;

  writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
  writeDescriptorSet.descriptorCount = 1;
  vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
  return descriptorSet;
}

static void loadGltf(VkDevice device, VkPhysicalDevice physicalDevice,
                     VkQueue copyQueue, VkCommandPool commandPool,
                     VkSurfaceKHR surface, VkRenderPass renderPass,
                     GLFWwindow *window,
                     const std::string &vertexShaderCodePath,
                     const std::string &fragmentShaderCodePath) {
  const auto scene{readScene("/home/seth/Documents/CesiumMan.gltf")};
  std::vector<VulkanImage> vulkanImages;
  transform(
      scene.images.begin(), scene.images.end(), back_inserter(vulkanImages),
      [device, physicalDevice, commandPool, copyQueue](const Image &image) {
        return vulkanImage(image.buffer.data(), image.buffer.size(),
                           VK_FORMAT_R8G8B8A8_UNORM, image.width, image.height,
                           device, physicalDevice, commandPool, copyQueue);
      });

  std::vector<VulkanBufferWithMemory> jointsVulkanBuffersWithMemory;
  for (const auto &skin : scene.skins)
    if (!skin.inverseBindMatrices.empty()) {
      // Store inverse bind matrices for this skin in a shader storage
      // buffer object To keep this sample simple, we create a host visible
      // shader storage buffer

      const VkDeviceSize bufferSize{sizeof(glm::mat4) *
                                    skin.inverseBindMatrices.size()};
      vulkan_wrappers::Buffer jointsVulkanBuffer{
          device, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, bufferSize};
      auto memory{bufferMemory(device, physicalDevice,
                               jointsVulkanBuffer.buffer,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
      copy(device, memory.memory, skin.inverseBindMatrices.data(), bufferSize);
      // VK_CHECK_RESULT(skins[i].ssbo.map());
      jointsVulkanBuffersWithMemory.push_back(VulkanBufferWithMemory{
          std::move(jointsVulkanBuffer), std::move(memory)});
    }

  // Calculate initial pose
  for (const auto &node : scene.nodes)
    updateJoints(node.get(), device);

  const auto vertexBufferWithMemory{bufferWithMemory(
      device, physicalDevice, commandPool, copyQueue,
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      scene.vertexBuffer.data(),
      scene.vertexBuffer.size() * sizeof(AnimatingVertex))};

  const auto indexBufferWithMemory{bufferWithMemory(
      device, physicalDevice, commandPool, copyQueue,
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      scene.indexBuffer.data(), scene.indexBuffer.size() * sizeof(uint32_t))};

  vulkan_wrappers::Buffer animatingUniformBuffer{
      device, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      sizeof(AnimatingUniformBufferObject)};
  auto animatingUniformBufferMemory{
      bufferMemory(device, physicalDevice, animatingUniformBuffer.buffer,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
  VulkanBufferWithMemory animatingUniformBufferWithMemory{
      std::move(animatingUniformBuffer),
      std::move(animatingUniformBufferMemory)};

  /*
          This sample uses separate descriptor sets (and layouts) for the
     matrices and materials (textures)
  */

  VkDescriptorPoolSize uniformBufferDescriptorPoolSize{};
  uniformBufferDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uniformBufferDescriptorPoolSize.descriptorCount = 1;

  VkDescriptorPoolSize imageSamplerDescriptorPoolSize{};
  imageSamplerDescriptorPoolSize.type =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  imageSamplerDescriptorPoolSize.descriptorCount =
      static_cast<uint32_t>(vulkanImages.size());

  VkDescriptorPoolSize storageBufferDescriptorPoolSize{};
  storageBufferDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  storageBufferDescriptorPoolSize.descriptorCount =
      static_cast<uint32_t>(scene.skins.size());
  std::vector<VkDescriptorPoolSize> poolSizes = {
      uniformBufferDescriptorPoolSize,
      // One combined image sampler per material image/texture
      imageSamplerDescriptorPoolSize,
      // One ssbo per skin
      storageBufferDescriptorPoolSize};
  // Number of descriptor sets = One for the scene ubo + one per image + one per
  // skin
  const auto maxSetCount{
      std::accumulate(poolSizes.begin(), poolSizes.end(), 0U,
                      [](uint32_t count, const VkDescriptorPoolSize &size) {
                        return count + size.descriptorCount;
                      })};

  vulkan_wrappers::DescriptorPool descriptorPool{device, poolSizes,
                                                 maxSetCount};

  // // The pipeline layout uses three sets:
  // // Set 0 = Scene matrices (VS)
  // // Set 1 = Joint matrices (VS)
  // // Set 2 = Material texture (FS)

  const auto jointMatricesSetLayout{
      animatingJointMatricesDescriptorSetLayout(device)};

  // Descriptor set for glTF model skin joint matrices
  std::vector<VkDescriptorSet> skinDescriptorSets;
  transform(
      jointsVulkanBuffersWithMemory.begin(),
      jointsVulkanBuffersWithMemory.end(), back_inserter(skinDescriptorSets),
      [&device, &descriptorPool,
       &jointMatricesSetLayout](const VulkanBufferWithMemory &buffer) {
        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
        descriptorSetAllocateInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.descriptorPool =
            descriptorPool.descriptorPool;
        descriptorSetAllocateInfo.pSetLayouts =
            &jointMatricesSetLayout.descriptorSetLayout;
        descriptorSetAllocateInfo.descriptorSetCount = 1;

        VkDescriptorSet descriptorSet = nullptr;
        vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo,
                                 &descriptorSet);

        VkWriteDescriptorSet writeDescriptorSet{};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet = descriptorSet;
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSet.dstBinding = 0;

        VkDescriptorBufferInfo descriptorBufferInfo;
        descriptorBufferInfo.buffer = buffer.buffer.buffer;
        descriptorBufferInfo.offset = 0;
        descriptorBufferInfo.range = VK_WHOLE_SIZE;

        writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
        writeDescriptorSet.descriptorCount = 1;
        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
        return descriptorSet;
      });

  const auto samplerSetLayout{animatingSamplerDescriptorSetLayout(device)};

  // Descriptor sets for glTF model materials
  const vulkan_wrappers::Sampler vulkanTextureSampler{device, physicalDevice,
                                                      1};
  std::vector<VkDescriptorSet> textureDescriptorSets;
  transform(vulkanImages.begin(), vulkanImages.end(),
            back_inserter(skinDescriptorSets),
            [&device, &descriptorPool, &samplerSetLayout,
             &vulkanTextureSampler](const VulkanImage &image) {
              VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
              descriptorSetAllocateInfo.sType =
                  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
              descriptorSetAllocateInfo.descriptorPool =
                  descriptorPool.descriptorPool;
              descriptorSetAllocateInfo.pSetLayouts =
                  &samplerSetLayout.descriptorSetLayout;
              descriptorSetAllocateInfo.descriptorSetCount = 1;
              VkDescriptorSet descriptorSet = nullptr;
              vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo,
                                       &descriptorSet);

              VkWriteDescriptorSet writeDescriptorSet{};
              writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
              writeDescriptorSet.dstSet = descriptorSet;
              writeDescriptorSet.descriptorType =
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
              writeDescriptorSet.dstBinding = 0;

              VkDescriptorImageInfo descriptor;
              descriptor.sampler = vulkanTextureSampler.sampler;
              descriptor.imageView = image.view.view;
              descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

              writeDescriptorSet.pImageInfo = &descriptor;
              writeDescriptorSet.descriptorCount = 1;
              vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0,
                                     nullptr);
              return descriptorSet;
            });

  const auto uboSetLayout{
      animatingUniformBufferObjectDescriptorSetLayout(device)};

  // We will use push constants to push the local matrices of a primitive to the
  // vertex shader
  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(glm::mat4);

  const vulkan_wrappers::PipelineLayout vulkanPipelineLayout{
      device,
      {uboSetLayout.descriptorSetLayout, samplerSetLayout.descriptorSetLayout,
       jointMatricesSetLayout.descriptorSetLayout},
      {pushConstantRange}};

  // Descriptor set for scene matrices
  const auto uboDescriptorSet{animatingUniformBufferObjectDescriptorSet(
      device, animatingUniformBufferWithMemory, descriptorPool, uboSetLayout)};

  const auto pipeline{animatingPipeline(
      device, physicalDevice, surface, renderPass, window, vertexShaderCodePath,
      fragmentShaderCodePath, vulkanPipelineLayout)};
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
