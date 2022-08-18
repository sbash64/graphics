#include <sbash64/graphics/glfw-wrappers.hpp>
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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <stack>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace sbash64::graphics {
constexpr auto clamp(int velocity, int limit) -> int {
  return std::clamp(velocity, -limit, limit);
}

constexpr auto withFriction(int_fast64_t velocity, int_fast64_t friction)
    -> int_fast64_t {
  return (velocity < 0 ? -1 : 1) *
         std::max(int_fast64_t{0}, std::abs(velocity) - friction);
}

constexpr std::array<int, 91> sinTimesMillionLUT = {
    0,      17452,  34899,  52336,  69756,  87156,  104528, 121869, 139173,
    156434, 173648, 190809, 207912, 224951, 241922, 258819, 275637, 292372,
    309017, 325568, 342020, 358368, 374607, 390731, 406737, 422618, 438371,
    453990, 469472, 484810, 500000, 515038, 529919, 544639, 559193, 573576,
    587785, 601815, 615661, 629320, 642788, 656059, 669131, 681998, 694658,
    707107, 719340, 731354, 743145, 754710, 766044, 777146, 788011, 798636,
    809017, 819152, 829038, 838671, 848048, 857167, 866025, 874620, 882948,
    891007, 898794, 906308, 913545, 920505, 927184, 933580, 939693, 945519,
    951057, 956305, 961262, 965926, 970296, 974370, 978148, 981627, 984808,
    987688, 990268, 992546, 994522, 996195, 997564, 998630, 999391, 999848,
    1000000};

constexpr auto sinTimesMillion(int degrees) -> int {
  if (degrees < 0)
    return -sinTimesMillion(-degrees);
  if (degrees < 90)
    return sinTimesMillionLUT.at(degrees);
  if (degrees < 180)
    return sinTimesMillionLUT.at(180 - degrees);
  if (degrees < 270)
    return -sinTimesMillionLUT.at(degrees - 180);
  if (degrees < 360)
    return -sinTimesMillionLUT.at(360 - degrees);
  return 0;
}

static_assert(sinTimesMillion(0) == 0, "sin implementation incorrect");
static_assert(sinTimesMillion(2) == 34899, "sin implementation incorrect");
static_assert(sinTimesMillion(89) == 999848, "sin implementation incorrect");
static_assert(sinTimesMillion(90) == 1000000, "sin implementation incorrect");
static_assert(sinTimesMillion(91) == 999848, "sin implementation incorrect");
static_assert(sinTimesMillion(179) == 17452, "sin implementation incorrect");
static_assert(sinTimesMillion(180) == 0, "sin implementation incorrect");
static_assert(sinTimesMillion(181) == -17452, "sin implementation incorrect");
static_assert(sinTimesMillion(269) == -999848, "sin implementation incorrect");
static_assert(sinTimesMillion(270) == -1000000, "sin implementation incorrect");
static_assert(sinTimesMillion(271) == -999848, "sin implementation incorrect");
static_assert(sinTimesMillion(359) == -17452, "sin implementation incorrect");
static_assert(sinTimesMillion(360) == 0, "sin implementation incorrect");
static_assert(sinTimesMillion(-10) == -173648, "sin implementation incorrect");

constexpr auto cosTimesMillion(int degrees) -> int {
  if (degrees < 0)
    return cosTimesMillion(-degrees);
  if (degrees < 90)
    return sinTimesMillionLUT.at(90 - degrees);
  if (degrees < 180)
    return -sinTimesMillionLUT.at(degrees - 90);
  if (degrees < 270)
    return -sinTimesMillionLUT.at(270 - degrees);
  if (degrees < 360)
    return sinTimesMillionLUT.at(degrees - 270);
  return 1000000;
}

static_assert(cosTimesMillion(0) == 1000000, "cos implementation incorrect");
static_assert(cosTimesMillion(1) == 999848, "cos implementation incorrect");
static_assert(cosTimesMillion(89) == 17452, "cos implementation incorrect");
static_assert(cosTimesMillion(90) == 0, "cos implementation incorrect");
static_assert(cosTimesMillion(91) == -17452, "cos implementation incorrect");
static_assert(cosTimesMillion(179) == -999848, "cos implementation incorrect");
static_assert(cosTimesMillion(180) == -1000000, "cos implementation incorrect");
static_assert(cosTimesMillion(181) == -999848, "cos implementation incorrect");
static_assert(cosTimesMillion(269) == -17452, "cos implementation incorrect");
static_assert(cosTimesMillion(270) == 0, "cos implementation incorrect");
static_assert(cosTimesMillion(271) == 17452, "cos implementation incorrect");
static_assert(cosTimesMillion(359) == 999848, "cos implementation incorrect");
static_assert(cosTimesMillion(360) == 1000000, "cos implementation incorrect");

enum class JumpState { grounded, started, released };

constexpr auto absoluteValue(int a) -> int { return a < 0 ? -a : a; }

struct ScreenPoint {
  double x;
  double y;
};

struct Mouse {
  ScreenPoint position;
  bool leftPressed;
  bool rightPressed;
  bool middlePressed;
};

struct Camera {
  int yawDegrees;
  int pitchDegrees;
};

struct IntegerVector3D {
  int_fast64_t x;
  int_fast64_t y;
  int_fast64_t z;
};

struct GlfwCallback {
  Mouse mouse{};
  bool frameBufferResized{};
};

static void updateMousePosition(GlfwCallback *callback, double x, double y) {
  callback->mouse.position = {x, y};
}

static void onCursorPositionChanged(GLFWwindow *window, double x, double y) {
  updateMousePosition(
      static_cast<GlfwCallback *>(glfwGetWindowUserPointer(window)), x, y);
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

static auto pressing(GLFWwindow *window, int key) -> bool {
  return glfwGetKey(window, key) == GLFW_PRESS;
}

static void onFramebufferResize(GLFWwindow *window, int /*width*/,
                                int /*height*/) {
  auto *const glfwCallback =
      static_cast<GlfwCallback *>(glfwGetWindowUserPointer(window));
  glfwCallback->frameBufferResized = true;
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

struct VulkanDrawable {
  VulkanBufferWithMemory vertexBufferWithMemory;
  VulkanBufferWithMemory indexBufferWithMemory;
};

static auto textureImage(const void *buffer, VkDeviceSize bufferSize,
                         VkFormat format, uint32_t width, uint32_t height,
                         VkDevice device, VkPhysicalDevice physicalDevice,
                         VkCommandPool commandPool, VkQueue queue)
    -> VulkanImage {
  const auto mipLevels{
      static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) +
      1};
  vulkan_wrappers::Image image{
      device,
      width,
      height,
      format,
      VK_IMAGE_TILING_OPTIMAL,
      static_cast<uint32_t>(VK_IMAGE_USAGE_TRANSFER_SRC_BIT) |
          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      mipLevels,
      VK_SAMPLE_COUNT_1_BIT};
  auto memory{imageMemory(device, physicalDevice, image.image,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
  transitionImageLayout(device, commandPool, queue, image.image,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
  const vulkan_wrappers::Buffer stagingBuffer{
      device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, bufferSize};
  const auto stagingMemory{
      bufferMemory(device, physicalDevice, stagingBuffer.buffer,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
  copy(device, stagingMemory.memory, buffer, bufferSize);
  copyBufferToImage(device, commandPool, queue, stagingBuffer.buffer,
                    image.image, width, height);
  generateMipmaps(device, physicalDevice, commandPool, queue, image.image,
                  format, width, height, mipLevels);
  vulkan_wrappers::ImageView view{device, image.image, format,
                                  VK_IMAGE_ASPECT_COLOR_BIT, mipLevels};
  return VulkanImage{std::move(image), std::move(view), std::move(memory)};
}

static auto textureImages(VkDevice device, VkPhysicalDevice physicalDevice,
                          VkCommandPool commandPool, VkQueue queue,
                          const Scene &scene) -> std::vector<VulkanImage> {
  std::vector<VulkanImage> vulkanImages;
  transform(
      scene.images.begin(), scene.images.end(), back_inserter(vulkanImages),
      [device, physicalDevice, commandPool, queue](const Image &image) {
        return textureImage(image.buffer.data(), image.buffer.size(),
                            VK_FORMAT_R8G8B8A8_UNORM, image.width, image.height,
                            device, physicalDevice, commandPool, queue);
      });
  return vulkanImages;
}

static auto uniformBufferWithMemory(VkDevice device,
                                    VkPhysicalDevice physicalDevice,
                                    VkDeviceSize bufferSize)
    -> VulkanBufferWithMemory {
  vulkan_wrappers::Buffer buffer{device, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                 bufferSize};
  auto memory{bufferMemory(device, physicalDevice, buffer.buffer,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
  return VulkanBufferWithMemory{std::move(buffer), std::move(memory)};
}

static auto getLocalMatrix(const Node &node) -> glm::mat4 {
  return glm::translate(glm::mat4(1.0F), node.translation) *
         glm::mat4(node.rotation) * glm::scale(glm::mat4(1.0F), node.scale) *
         node.matrix;
}

static auto getNodeMatrix(const Node *node) -> glm::mat4 {
  auto nodeMatrix{getLocalMatrix(*node)};
  auto *currentParent = node->parent;
  while (currentParent != nullptr) {
    nodeMatrix = getLocalMatrix(*currentParent) * nodeMatrix;
    currentParent = currentParent->parent;
  }
  return nodeMatrix;
}

static void
draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout,
     const std::vector<VkDescriptorSet> &textureDescriptorSets,
     uint32_t textureDescriptorSetNumber, const Node &root, const Scene &scene,
     const std::function<void(VkCommandBuffer, VkPipelineLayout, const Node &)>
         &perMeshBind = {}) {
  std::stack<const Node *> stack{};
  stack.push(&root);
  while (!stack.empty()) {
    const auto *const current{stack.top()};
    stack.pop();
    if (!current->mesh.primitives.empty()) {
      auto nodeMatrix{current->matrix};
      const auto *currentParent = current->parent;
      while (currentParent != nullptr) {
        nodeMatrix = currentParent->matrix * nodeMatrix;
        currentParent = currentParent->parent;
      }
      vkCmdPushConstants(commandBuffer, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(nodeMatrix),
                         &nodeMatrix);
      if (perMeshBind)
        perMeshBind(commandBuffer, pipelineLayout, *current);
      for (const auto &primitive : current->mesh.primitives)
        if (primitive.indexCount > 0) {
          std::array<VkDescriptorSet, 1> descriptorSets{
              textureDescriptorSets.at(scene.textureIndices.at(
                  scene.materials.at(primitive.materialIndex)
                      .baseColorTextureIndex))};
          vkCmdBindDescriptorSets(
              commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
              textureDescriptorSetNumber, descriptorSets.size(),
              descriptorSets.data(), 0, nullptr);
          vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1,
                           primitive.firstIndex, 0, 0);
        }
    }
    for (const auto &child : current->children)
      stack.push(child.get());
  }
}

static void updateJointMatricesStorageBuffers(
    const Node *root, VkDevice device,
    const std::map<int, VulkanBufferWithMemory> &jointMatricesStorageBuffers,
    const Scene &scene) {
  std::stack<const Node *> stack{};
  stack.push(root);
  while (!stack.empty()) {
    const auto *const current{stack.top()};
    stack.pop();
    if (current->skin > -1) {
      const auto inverseTransform{glm::inverse(getNodeMatrix(current))};
      const auto joints{scene.skins.at(current->skin).joints};
      std::vector<glm::mat4> jointMatrices(joints.size());
      for (auto i{0}; i < jointMatrices.size(); i++)
        jointMatrices.at(i) =
            inverseTransform * getNodeMatrix(joints.at(i)) *
            scene.skins.at(current->skin).inverseBindMatrices.at(i);
      copy(device, jointMatricesStorageBuffers.at(current->skin).memory.memory,
           jointMatrices.data(),
           jointMatrices.size() * sizeof(decltype(jointMatrices)::value_type));
    }
    for (const auto &child : current->children)
      stack.push(child.get());
  }
}

static void updateModelViewProjectionUniformBuffer(
    const vulkan_wrappers::Device &device,
    const vulkan_wrappers::DeviceMemory &memory, glm::mat4 view,
    glm::mat4 perspective, glm::vec3 translation, float scale = 1,
    float rotationXAngleDegrees = 0, float rotationZAngleDegrees = 0) {
  UniformBufferObject ubo{};
  perspective[1][1] *= -1;
  ubo.mvp =
      perspective * view * glm::translate(glm::mat4{1.F}, translation) *
      glm::rotate(glm::mat4{1.F}, glm::radians(rotationXAngleDegrees),
                  glm::vec3{1.F, 0.F, 0.F}) *
      glm::rotate(glm::mat4{1.F}, glm::radians(0.F), glm::vec3{0.F, 1.F, 0.F}) *
      glm::rotate(glm::mat4{1.F}, glm::radians(rotationZAngleDegrees),
                  glm::vec3{0.F, 0.F, 1.F}) *
      glm::scale(glm::vec3{scale, scale, scale});
  copy(device.device, memory.memory, &ubo, sizeof(ubo));
}

static auto quaternion(glm::vec4 v) -> glm::quat {
  glm::quat quat;
  quat.x = v.x;
  quat.y = v.y;
  quat.z = v.z;
  quat.w = v.w;
  return quat;
}

static auto jointMatricesStorageBuffers(VkDevice device,
                                        VkPhysicalDevice physicalDevice,
                                        const Scene &scene)
    -> std::map<int, VulkanBufferWithMemory> {
  std::map<int, VulkanBufferWithMemory> buffersWithMemory;
  int index{0};
  for (const auto &skin : scene.skins)
    if (!skin.inverseBindMatrices.empty()) {
      const VkDeviceSize bufferSize{sizeof(glm::mat4) *
                                    skin.inverseBindMatrices.size()};
      vulkan_wrappers::Buffer buffer{device, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     bufferSize};
      auto memory{bufferMemory(device, physicalDevice, buffer.buffer,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
      copy(device, memory.memory, skin.inverseBindMatrices.data(), bufferSize);
      buffersWithMemory.emplace(
          index, VulkanBufferWithMemory{std::move(buffer), std::move(memory)});
      ++index;
    }
  return buffersWithMemory;
}

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

static auto
animatingDescriptorPool(VkDevice device,
                        const std::vector<VulkanImage> &textureImages,
                        const Scene &scene) -> vulkan_wrappers::DescriptorPool {
  VkDescriptorPoolSize uniformBufferDescriptorPoolSize{};
  uniformBufferDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uniformBufferDescriptorPoolSize.descriptorCount = 1;

  VkDescriptorPoolSize imageSamplerDescriptorPoolSize{};
  imageSamplerDescriptorPoolSize.type =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  imageSamplerDescriptorPoolSize.descriptorCount =
      static_cast<uint32_t>(textureImages.size());

  VkDescriptorPoolSize storageBufferDescriptorPoolSize{};
  storageBufferDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  storageBufferDescriptorPoolSize.descriptorCount =
      static_cast<uint32_t>(scene.skins.size());
  std::vector<VkDescriptorPoolSize> poolSizes = {
      uniformBufferDescriptorPoolSize, imageSamplerDescriptorPoolSize,
      storageBufferDescriptorPoolSize};
  const auto maxSetCount{
      std::accumulate(poolSizes.begin(), poolSizes.end(), 0U,
                      [](uint32_t count, const VkDescriptorPoolSize &size) {
                        return count + size.descriptorCount;
                      })};
  return {device, poolSizes, maxSetCount};
}

static auto
stationaryDescriptorPool(VkDevice device,
                         const std::vector<VulkanImage> &textureImages)
    -> vulkan_wrappers::DescriptorPool {
  VkDescriptorPoolSize uniformBufferDescriptorPoolSize{};
  uniformBufferDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uniformBufferDescriptorPoolSize.descriptorCount = 1;

  VkDescriptorPoolSize imageSamplerDescriptorPoolSize{};
  imageSamplerDescriptorPoolSize.type =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  imageSamplerDescriptorPoolSize.descriptorCount =
      static_cast<uint32_t>(textureImages.size()) + 100;

  std::vector<VkDescriptorPoolSize> poolSizes = {
      uniformBufferDescriptorPoolSize, imageSamplerDescriptorPoolSize};
  const auto maxSetCount{
      std::accumulate(poolSizes.begin(), poolSizes.end(), 0U,
                      [](uint32_t count, const VkDescriptorPoolSize &size) {
                        return count + size.descriptorCount;
                      })};
  return {device, poolSizes, maxSetCount + 100};
}

static auto jointMatricesStorageBufferDescriptorSets(
    VkDevice device, const vulkan_wrappers::DescriptorPool &descriptorPool,
    const vulkan_wrappers::DescriptorSetLayout &setLayout,
    const std::map<int, VulkanBufferWithMemory> &buffersWithMemory)
    -> std::map<int, VkDescriptorSet> {
  std::map<int, VkDescriptorSet> jointMatricesDescriptorSets;
  transform(
      buffersWithMemory.begin(), buffersWithMemory.end(),
      inserter(jointMatricesDescriptorSets,
               jointMatricesDescriptorSets.begin()),
      [&device, &descriptorPool,
       &setLayout](const std::pair<const int, VulkanBufferWithMemory> &buffer) {
        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
        descriptorSetAllocateInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.descriptorPool =
            descriptorPool.descriptorPool;
        descriptorSetAllocateInfo.pSetLayouts = &setLayout.descriptorSetLayout;
        descriptorSetAllocateInfo.descriptorSetCount = 1;

        VkDescriptorSet descriptorSet{nullptr};
        throwOnError(
            [&descriptorSetAllocateInfo, &descriptorSet, device]() {
              // deallocated by descriptorPool
              return vkAllocateDescriptorSets(
                  device, &descriptorSetAllocateInfo, &descriptorSet);
            },
            "failed to allocate descriptor sets!");

        VkWriteDescriptorSet writeDescriptorSet{};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet = descriptorSet;
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSet.dstBinding = 0;

        VkDescriptorBufferInfo descriptorBufferInfo;
        descriptorBufferInfo.buffer = buffer.second.buffer.buffer;
        descriptorBufferInfo.offset = 0;
        descriptorBufferInfo.range = VK_WHOLE_SIZE;

        writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
        writeDescriptorSet.descriptorCount = 1;
        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
        return std::make_pair(buffer.first, descriptorSet);
      });
  return jointMatricesDescriptorSets;
}

static auto combinedImageSamplerDescriptorSets(
    VkDescriptorPool descriptorPool, const vulkan_wrappers::Device &device,
    const vulkan_wrappers::Sampler &sampler,
    const vulkan_wrappers::DescriptorSetLayout &descriptorSetLayout,
    const std::vector<VulkanImage> &images) -> std::vector<VkDescriptorSet> {
  std::vector<VkDescriptorSet> sets;
  transform(images.begin(), images.end(), back_inserter(sets),
            [descriptorPool, &device, &sampler,
             &descriptorSetLayout](const VulkanImage &image) {
              VkDescriptorSet descriptorSet{nullptr};

              VkDescriptorSetAllocateInfo allocInfo{};
              allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
              allocInfo.descriptorPool = descriptorPool;
              allocInfo.descriptorSetCount = 1;
              allocInfo.pSetLayouts = &descriptorSetLayout.descriptorSetLayout;

              throwOnError(
                  [&]() {
                    // deallocated by descriptorPool
                    return vkAllocateDescriptorSets(device.device, &allocInfo,
                                                    &descriptorSet);
                  },
                  "failed to allocate descriptor sets!");

              VkDescriptorImageInfo imageInfo{};
              imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
              imageInfo.imageView = image.view.view;
              imageInfo.sampler = sampler.sampler;

              std::array<VkWriteDescriptorSet, 1> descriptorWrite{};

              descriptorWrite.at(0).sType =
                  VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
              descriptorWrite.at(0).dstSet = descriptorSet;
              descriptorWrite.at(0).dstBinding = 0;
              descriptorWrite.at(0).dstArrayElement = 0;
              descriptorWrite.at(0).descriptorType =
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
              descriptorWrite.at(0).descriptorCount = 1;
              descriptorWrite.at(0).pImageInfo = &imageInfo;

              vkUpdateDescriptorSets(device.device, descriptorWrite.size(),
                                     descriptorWrite.data(), 0, nullptr);
              return descriptorSet;
            });
  return sets;
}

static auto uniformBufferDescriptorSet(
    VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout,
    VkDevice device, VkBuffer buffer) -> VkDescriptorSet {
  VkDescriptorSet descriptorSet{nullptr};

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &descriptorSetLayout;

  throwOnError(
      [&]() {
        // freed by descriptorPool
        return vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);
      },
      "failed to allocate descriptor sets!");

  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = buffer;
  bufferInfo.offset = 0;
  bufferInfo.range = VK_WHOLE_SIZE;

  VkWriteDescriptorSet descriptorWrite{};
  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = descriptorSet;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pBufferInfo = &bufferInfo;

  vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
  return descriptorSet;
}

static auto horizontalSpeedSquaredOfVelocityVector(IntegerVector3D v)
    -> int_fast64_t {
  return v.x * v.x + v.z * v.z;
}

static auto applyVelocity(IntegerVector3D displacement,
                          IntegerVector3D velocity) -> IntegerVector3D {
  return {displacement.x + velocity.x, displacement.y + velocity.y,
          displacement.z + velocity.z};
}

// https://stackoverflow.com/a/18067292
static auto divide(int_fast64_t n, int_fast64_t d) -> int_fast64_t {
  return ((n < 0) ^ (d < 0)) != 0 ? ((n - d / 2) / d) : ((n + d / 2) / d);
}

static auto applyHorizontalAcceleration(IntegerVector3D velocity,
                                        int acceleration, Camera camera,
                                        int degreeOffset) -> IntegerVector3D {
  return {
      velocity.x + divide(static_cast<int_fast64_t>(acceleration) *
                              cosTimesMillion(camera.yawDegrees + degreeOffset),
                          1000000),
      velocity.y,
      velocity.z + divide(static_cast<int_fast64_t>(acceleration) *
                              sinTimesMillion(camera.yawDegrees + degreeOffset),
                          1000000)};
}

static auto updateCamera(Camera camera, const GlfwCallback &glfwCallback,
                         ScreenPoint lastMousePosition) -> Camera {
  const auto sensitivity{0.5};
  camera.yawDegrees += static_cast<int>(std::round(
      (glfwCallback.mouse.position.x - lastMousePosition.x) * sensitivity));
  if (camera.yawDegrees > 179)
    camera.yawDegrees -= 360;
  if (camera.yawDegrees < -179)
    camera.yawDegrees += 360;
  camera.pitchDegrees =
      std::clamp(camera.pitchDegrees +
                     static_cast<int>(std::round(
                         (glfwCallback.mouse.position.y - lastMousePosition.y) *
                         sensitivity)),
                 1, 179);
  return camera;
}

static void bindGraphics(VkDescriptorSet descriptorSet,
                         const vulkan_wrappers::PipelineLayout &pipelineLayout,
                         VkCommandBuffer commandBuffer) {
  std::array<VkDescriptorSet, 1> descriptorSets{descriptorSet};
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelineLayout.pipelineLayout, 0,
                          descriptorSets.size(), descriptorSets.data(), 0,
                          nullptr);
}

static void bindVertex(const VulkanBufferWithMemory &bufferWithMemory,
                       VkCommandBuffer commandBuffer) {
  std::array<VkDeviceSize, 1> offsets{0};
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, &bufferWithMemory.buffer.buffer,
                         offsets.data());
}

static void run(const std::string &stationaryVertexShaderCodePath,
                const std::string &stationaryFragmentShaderCodePath,
                const std::string &animatingVertexShaderCodePath,
                const std::string &animatingFragmentShaderCodePath,
                const std::string &worldScenePath,
                const std::string &playerScenePath) {
  const glfw_wrappers::Init glfwInitialization;
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  const glfw_wrappers::Window glfwWindow{
      1280, 720}; // nintendo switch screen 1280 x 720
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
  const vulkan_wrappers::Sampler vulkanTextureSampler{vulkanDevice.device,
                                                      vulkanPhysicalDevice, 13};
  VkQueue graphicsQueue{nullptr};
  vkGetDeviceQueue(vulkanDevice.device,
                   graphicsSupportingQueueFamilyIndex(vulkanPhysicalDevice), 0,
                   &graphicsQueue);
  const vulkan_wrappers::CommandPool vulkanCommandPool{vulkanDevice.device,
                                                       vulkanPhysicalDevice};

  VkDescriptorSetLayoutBinding
      stationaryUniformBufferDescriptorSetLayoutBinding{};
  stationaryUniformBufferDescriptorSetLayoutBinding.binding = 0;
  stationaryUniformBufferDescriptorSetLayoutBinding.descriptorCount = 1;
  stationaryUniformBufferDescriptorSetLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  stationaryUniformBufferDescriptorSetLayoutBinding.pImmutableSamplers =
      nullptr;
  stationaryUniformBufferDescriptorSetLayoutBinding.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT;
  const vulkan_wrappers::DescriptorSetLayout
      stationaryUniformBufferDescriptorSetLayout{
          vulkanDevice.device,
          {stationaryUniformBufferDescriptorSetLayoutBinding}};

  VkDescriptorSetLayoutBinding
      stationaryCombinedImageSamplerDescriptorSetLayoutBinding{};
  stationaryCombinedImageSamplerDescriptorSetLayoutBinding.binding = 0;
  stationaryCombinedImageSamplerDescriptorSetLayoutBinding.descriptorCount = 1;
  stationaryCombinedImageSamplerDescriptorSetLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  stationaryCombinedImageSamplerDescriptorSetLayoutBinding.pImmutableSamplers =
      nullptr;
  stationaryCombinedImageSamplerDescriptorSetLayoutBinding.stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT;
  const vulkan_wrappers::DescriptorSetLayout
      stationaryCombinedImageSamplerDescriptorSetLayout{
          vulkanDevice.device,
          {stationaryCombinedImageSamplerDescriptorSetLayoutBinding}};

  const auto worldScene{readScene(worldScenePath)};
  const auto worldTextureImages{
      textureImages(vulkanDevice.device, vulkanPhysicalDevice,
                    vulkanCommandPool.commandPool, graphicsQueue, worldScene)};

  const auto worldDescriptorPool{
      stationaryDescriptorPool(vulkanDevice.device, worldTextureImages)};

  const auto worldCombinedImageSamplerDescriptorSets{
      graphics::combinedImageSamplerDescriptorSets(
          worldDescriptorPool.descriptorPool, vulkanDevice,
          vulkanTextureSampler,
          stationaryCombinedImageSamplerDescriptorSetLayout,
          worldTextureImages)};

  const auto worldVertexBuffer{bufferWithMemory(
      vulkanDevice.device, vulkanPhysicalDevice, vulkanCommandPool.commandPool,
      graphicsQueue,
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      worldScene.vertices.data(),
      worldScene.vertices.size() *
          sizeof(decltype(worldScene.vertices)::value_type))};

  const auto worldIndexBuffer{bufferWithMemory(
      vulkanDevice.device, vulkanPhysicalDevice, vulkanCommandPool.commandPool,
      graphicsQueue,
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      worldScene.vertexIndices.data(),
      worldScene.vertexIndices.size() *
          sizeof(decltype(worldScene.vertexIndices)::value_type))};

  const auto worldModelViewProjectionUniformBuffer{uniformBufferWithMemory(
      vulkanDevice.device, vulkanPhysicalDevice, sizeof(UniformBufferObject))};

  auto *const worldModelViewProjectionUniformBufferDescriptorSet{
      uniformBufferDescriptorSet(
          worldDescriptorPool.descriptorPool,
          stationaryUniformBufferDescriptorSetLayout.descriptorSetLayout,
          vulkanDevice.device,
          worldModelViewProjectionUniformBuffer.buffer.buffer)};

  const auto worldTextureImageDescriptors{combinedImageSamplerDescriptorSets(
      worldDescriptorPool.descriptorPool, vulkanDevice, vulkanTextureSampler,
      stationaryCombinedImageSamplerDescriptorSetLayout, worldTextureImages)};

  const vulkan_wrappers::CommandBuffers vulkanDrawCommandBuffers{
      vulkanDevice.device, vulkanCommandPool.commandPool,
      vulkanFrameBuffers.size()};

  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(glm::mat4);
  const vulkan_wrappers::PipelineLayout stationaryPipelineLayout{
      vulkanDevice.device,
      {stationaryUniformBufferDescriptorSetLayout.descriptorSetLayout,
       stationaryCombinedImageSamplerDescriptorSetLayout.descriptorSetLayout},
      {pushConstantRange}};

  std::vector<VkVertexInputAttributeDescription> attributeDescriptions(2);
  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(AnimatingVertex, pos);

  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(AnimatingVertex, uv);

  std::vector<VkVertexInputBindingDescription> bindingDescription(1);
  bindingDescription[0].binding = 0;
  bindingDescription[0].stride = sizeof(AnimatingVertex);
  bindingDescription[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  const vulkan_wrappers::Pipeline stationaryVulkanPipeline{
      vulkanDevice.device,
      vulkanPhysicalDevice,
      vulkanSurface.surface,
      stationaryPipelineLayout.pipelineLayout,
      vulkanRenderPass.renderPass,
      stationaryVertexShaderCodePath,
      stationaryFragmentShaderCodePath,
      glfwWindow.window,
      attributeDescriptions,
      bindingDescription};

  const auto playerScene{readScene(playerScenePath)};
  const auto playerTextureImages{
      textureImages(vulkanDevice.device, vulkanPhysicalDevice,
                    vulkanCommandPool.commandPool, graphicsQueue, playerScene)};

  const auto animatingDescriptorPool{graphics::animatingDescriptorPool(
      vulkanDevice.device, playerTextureImages, playerScene)};

  const auto playerJointMatricesStorageBuffers{
      graphics::jointMatricesStorageBuffers(vulkanDevice.device,
                                            vulkanPhysicalDevice, playerScene)};

  VkDescriptorSetLayoutBinding
      animatingUniformBufferDescriptorSetLayoutBinding{};
  animatingUniformBufferDescriptorSetLayoutBinding.binding = 0;
  animatingUniformBufferDescriptorSetLayoutBinding.descriptorCount = 1;
  animatingUniformBufferDescriptorSetLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  animatingUniformBufferDescriptorSetLayoutBinding.pImmutableSamplers = nullptr;
  animatingUniformBufferDescriptorSetLayoutBinding.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT;
  const vulkan_wrappers::DescriptorSetLayout
      animatingUniformBufferDescriptorSetLayout{
          vulkanDevice.device,
          {animatingUniformBufferDescriptorSetLayoutBinding}};

  VkDescriptorSetLayoutBinding jointMatricesDescriptorSetLayoutBinding{};
  jointMatricesDescriptorSetLayoutBinding.binding = 0;
  jointMatricesDescriptorSetLayoutBinding.descriptorCount = 1;
  jointMatricesDescriptorSetLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  jointMatricesDescriptorSetLayoutBinding.pImmutableSamplers = nullptr;
  jointMatricesDescriptorSetLayoutBinding.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT;
  const vulkan_wrappers::DescriptorSetLayout jointMatricesDescriptorSetLayout{
      vulkanDevice.device, {jointMatricesDescriptorSetLayoutBinding}};

  VkDescriptorSetLayoutBinding
      animatingCombinedImageSamplerDescriptorSetLayoutBinding{};
  animatingCombinedImageSamplerDescriptorSetLayoutBinding.binding = 0;
  animatingCombinedImageSamplerDescriptorSetLayoutBinding.descriptorCount = 1;
  animatingCombinedImageSamplerDescriptorSetLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  animatingCombinedImageSamplerDescriptorSetLayoutBinding.pImmutableSamplers =
      nullptr;
  animatingCombinedImageSamplerDescriptorSetLayoutBinding.stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT;
  const vulkan_wrappers::DescriptorSetLayout
      animatingCombinedImageSamplerDescriptorSetLayout{
          vulkanDevice.device,
          {animatingCombinedImageSamplerDescriptorSetLayoutBinding}};

  const auto playerJointMatricesStorageBufferDescriptorSets{
      graphics::jointMatricesStorageBufferDescriptorSets(
          vulkanDevice.device, animatingDescriptorPool,
          jointMatricesDescriptorSetLayout, playerJointMatricesStorageBuffers)};

  const auto playerCombinedImageSamplerDescriptorSets{
      graphics::combinedImageSamplerDescriptorSets(
          animatingDescriptorPool.descriptorPool, vulkanDevice,
          vulkanTextureSampler,
          animatingCombinedImageSamplerDescriptorSetLayout,
          playerTextureImages)};

  const vulkan_wrappers::PipelineLayout animatingPipelineLayout{
      vulkanDevice.device,
      {animatingUniformBufferDescriptorSetLayout.descriptorSetLayout,
       jointMatricesDescriptorSetLayout.descriptorSetLayout,
       animatingCombinedImageSamplerDescriptorSetLayout.descriptorSetLayout},
      {pushConstantRange}};

  const auto playerVertexBuffer{bufferWithMemory(
      vulkanDevice.device, vulkanPhysicalDevice, vulkanCommandPool.commandPool,
      graphicsQueue,
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      playerScene.vertices.data(),
      playerScene.vertices.size() *
          sizeof(decltype(playerScene.vertices)::value_type))};

  const auto playerIndexBuffer{bufferWithMemory(
      vulkanDevice.device, vulkanPhysicalDevice, vulkanCommandPool.commandPool,
      graphicsQueue,
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      playerScene.vertexIndices.data(),
      playerScene.vertexIndices.size() *
          sizeof(decltype(playerScene.vertexIndices)::value_type))};

  const auto playerModelViewProjectionUniformBuffer{uniformBufferWithMemory(
      vulkanDevice.device, vulkanPhysicalDevice, sizeof(UniformBufferObject))};
  auto *const playerModelViewProjectionUniformBufferDescriptorSet{
      uniformBufferDescriptorSet(
          animatingDescriptorPool.descriptorPool,
          animatingUniformBufferDescriptorSetLayout.descriptorSetLayout,
          vulkanDevice.device,
          playerModelViewProjectionUniformBuffer.buffer.buffer)};

  const auto animatingPipeline{graphics::animatingPipeline(
      vulkanDevice.device, vulkanPhysicalDevice, vulkanSurface.surface,
      vulkanRenderPass.renderPass, glfwWindow.window,
      animatingVertexShaderCodePath, animatingFragmentShaderCodePath,
      animatingPipelineLayout)};

  auto frameBufferCount{0U};
  for (auto *const commandBuffer : vulkanDrawCommandBuffers.commandBuffers) {
    throwIfFailsToBegin(commandBuffer);
    beginRenderPass(vulkanPhysicalDevice, vulkanSurface.surface,
                    vulkanFrameBuffers.at(frameBufferCount).framebuffer,
                    commandBuffer, glfwWindow.window,
                    vulkanRenderPass.renderPass);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      stationaryVulkanPipeline.pipeline);
    bindGraphics(worldModelViewProjectionUniformBufferDescriptorSet,
                 stationaryPipelineLayout, commandBuffer);
    bindVertex(worldVertexBuffer, commandBuffer);
    vkCmdBindIndexBuffer(commandBuffer, worldIndexBuffer.buffer.buffer, 0,
                         VK_INDEX_TYPE_UINT32);
    for (const auto &node : worldScene.nodes)
      draw(commandBuffer, stationaryPipelineLayout.pipelineLayout,
           worldCombinedImageSamplerDescriptorSets, 1U, *node, worldScene);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      animatingPipeline.pipeline);
    bindGraphics(playerModelViewProjectionUniformBufferDescriptorSet,
                 animatingPipelineLayout, commandBuffer);
    bindVertex(playerVertexBuffer, commandBuffer);
    vkCmdBindIndexBuffer(commandBuffer, playerIndexBuffer.buffer.buffer, 0,
                         VK_INDEX_TYPE_UINT32);
    for (const auto &node : playerScene.nodes)
      draw(commandBuffer, animatingPipelineLayout.pipelineLayout,
           playerCombinedImageSamplerDescriptorSets, 2U, *node, playerScene,
           [&playerJointMatricesStorageBufferDescriptorSets](
               VkCommandBuffer commandBuffer_, VkPipelineLayout pipelineLayout,
               const Node &node_) {
             std::array<VkDescriptorSet, 1> descriptorSets{
                 playerJointMatricesStorageBufferDescriptorSets.at(node_.skin)};
             const auto set{1U};
             vkCmdBindDescriptorSets(commandBuffer_,
                                     VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     pipelineLayout, set, descriptorSets.size(),
                                     descriptorSets.data(), 0, nullptr);
           });
    vkCmdEndRenderPass(commandBuffer);
    throwOnError([&]() { return vkEndCommandBuffer(commandBuffer); },
                 "failed to record command buffer!");
    ++frameBufferCount;
  }

  VkQueue presentQueue{nullptr};
  vkGetDeviceQueue(vulkanDevice.device,
                   presentSupportingQueueFamilyIndex(vulkanPhysicalDevice,
                                                     vulkanSurface.surface),
                   0, &presentQueue);
  const auto maxFramesInFlight{2};
  std::vector<vulkan_wrappers::Fence> inFlightVulkanFences;
  generate_n(back_inserter(inFlightVulkanFences), maxFramesInFlight,
             [&vulkanDevice]() {
               return vulkan_wrappers::Fence{vulkanDevice.device};
             });
  const auto imageAvailableVulkanSemaphores{
      semaphores(vulkanDevice.device, maxFramesInFlight)};
  const auto renderFinishedVulkanSemaphores{
      semaphores(vulkanDevice.device, maxFramesInFlight)};
  std::vector<VkFence> imagesInFlight(swapChainImages.size(), VK_NULL_HANDLE);
  const auto swapChainExtent{swapExtent(
      vulkanPhysicalDevice, vulkanSurface.surface, glfwWindow.window)};

  auto animationTime{0.F};

  auto recreatingSwapChain{false};
  auto currentFrame{0U};
  IntegerVector3D playerVelocity{};
  auto playerJumpState{JumpState::grounded};
  auto worldOrigin{glm::vec3{0.F, 0.F, 0.F}};
  Camera camera{};
  camera.yawDegrees = 90;
  camera.pitchDegrees = 15;
  IntegerVector3D playerDisplacement{0, 0, 0};
  auto animationIndex{0};
  ScreenPoint lastMousePosition{};
  while (!recreatingSwapChain) {
    if (glfwWindowShouldClose(glfwWindow.window) != 0)
      break;

    glfwPollEvents();

    if (glfwCallback.mouse.leftPressed)
      camera = updateCamera(camera, glfwCallback, lastMousePosition);
    lastMousePosition = glfwCallback.mouse.position;

    if (pressing(glfwWindow.window, GLFW_KEY_F) && animationIndex == 0) {
      animationIndex = 1;
      animationTime = playerScene.animations.at(animationIndex).start;
    }

    constexpr auto playerRunAcceleration{1000};
    if (pressing(glfwWindow.window, GLFW_KEY_A))
      playerVelocity = applyHorizontalAcceleration(
          playerVelocity, playerRunAcceleration, camera, -90);
    if (pressing(glfwWindow.window, GLFW_KEY_D))
      playerVelocity = applyHorizontalAcceleration(
          playerVelocity, playerRunAcceleration, camera, 90);
    if (pressing(glfwWindow.window, GLFW_KEY_W))
      playerVelocity = applyHorizontalAcceleration(
          playerVelocity, playerRunAcceleration, camera, 0);
    if (pressing(glfwWindow.window, GLFW_KEY_S))
      playerVelocity = applyHorizontalAcceleration(
          playerVelocity, playerRunAcceleration, camera, 180);

    constexpr auto playerJumpAcceleration{10000};
    if (pressing(glfwWindow.window, GLFW_KEY_SPACE) &&
        playerJumpState == JumpState::grounded) {
      playerJumpState = JumpState::started;
      playerVelocity.y += playerJumpAcceleration;
    }
    const auto gravity{-500};
    playerVelocity.y += gravity;
    if (glfwGetKey(glfwWindow.window, GLFW_KEY_SPACE) != GLFW_PRESS &&
        playerJumpState == JumpState::started) {
      playerJumpState = JumpState::released;
      if (playerVelocity.y > 0)
        playerVelocity.y = 0;
    }
    // constexpr auto playerMaxGroundSpeed{10000};
    // if (horizontalSpeedOfVelocityVector(playerVelocity) >
    //     playerMaxGroundSpeed)
    //   ;
    // constexpr auto groundFriction{500};
    // playerVelocity.x = withFriction(playerVelocity.x, groundFriction);
    // playerVelocity.z = withFriction(playerVelocity.z, groundFriction);
    playerDisplacement = applyVelocity(playerDisplacement, playerVelocity);
    if (playerDisplacement.y < 0) {
      playerJumpState = JumpState::grounded;
      playerDisplacement.y = 0;
      playerVelocity.y = 0;
    }

    const auto animation{playerScene.animations.at(animationIndex)};

    for (const auto &channel : animation.channels) {
      const auto &sampler = animation.samplers[channel.samplerIndex];
      if (sampler.interpolation != "LINEAR")
        continue;
      for (auto i{0U}; i < sampler.inputs.size() - 1; i++)
        if (animationTime >= sampler.inputs[i] &&
            animationTime <= sampler.inputs[i + 1]) {
          const auto a = (animationTime - sampler.inputs[i]) /
                         (sampler.inputs[i + 1] - sampler.inputs[i]);
          if (channel.path == "translation")
            channel.node->translation =
                glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], a);
          else if (channel.path == "rotation")
            channel.node->rotation = glm::normalize(
                glm::slerp(quaternion(sampler.outputsVec4[i]),
                           quaternion(sampler.outputsVec4[i + 1]), a));
          else if (channel.path == "scale")
            channel.node->scale =
                glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], a);
        }
    }

    animationTime += 1.f / 60;
    if (animationTime > animation.end) {
      animationIndex = 0;
      animationTime = playerScene.animations.at(animationIndex).start;
    }

    const auto projection =
        glm::perspective(glm::radians(45.F),
                         static_cast<float>(swapChainExtent.width) /
                             static_cast<float>(swapChainExtent.height),
                         10.F, 1000.F);
    const glm::vec3 playerPosition{
        static_cast<float>(playerDisplacement.x) * .0003F,
        static_cast<float>(playerDisplacement.y) * .0003F,
        static_cast<float>(playerDisplacement.z) * .0003F};
    const auto playerCameraFocus{playerPosition + glm::vec3{0, 15.F, 0}};
    const auto view = glm::lookAt(
        playerCameraFocus +
            60.F * glm::vec3{-std::cosf(glm::radians(
                                 static_cast<float>(camera.yawDegrees))) *
                                 std::cosf(glm::radians(
                                     static_cast<float>(camera.pitchDegrees))),
                             std::sinf(glm::radians(
                                 static_cast<float>(camera.pitchDegrees))),
                             -std::sinf(glm::radians(
                                 static_cast<float>(camera.yawDegrees))) *
                                 std::cosf(glm::radians(
                                     static_cast<float>(camera.pitchDegrees)))},
        playerCameraFocus, glm::vec3(0, 1, 0));

    // writes to uniform and storage buffers
    updateModelViewProjectionUniformBuffer(
        vulkanDevice, worldModelViewProjectionUniformBuffer.memory, view,
        projection, worldOrigin, 10.F);
    updateModelViewProjectionUniformBuffer(
        vulkanDevice, playerModelViewProjectionUniformBuffer.memory, view,
        projection, playerPosition, 2000.F, -90.F, 90.F - camera.yawDegrees);
    for (const auto &node : playerScene.nodes)
      updateJointMatricesStorageBuffers(node.get(), vulkanDevice.device,
                                        playerJointMatricesStorageBuffers,
                                        playerScene);
    // end writes

    vkWaitForFences(vulkanDevice.device, 1,
                    &inFlightVulkanFences[currentFrame].fence, VK_TRUE,
                    UINT64_MAX);

    uint32_t imageIndex{0};
    {
      const auto result{vkAcquireNextImageKHR(
          vulkanDevice.device, vulkanSwapchain.swapChain, UINT64_MAX,
          imageAvailableVulkanSemaphores[currentFrame].semaphore,
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

    imagesInFlight[imageIndex] = inFlightVulkanFences[currentFrame].fence;
    submit(vulkanDevice, graphicsQueue,
           imageAvailableVulkanSemaphores[currentFrame].semaphore,
           renderFinishedVulkanSemaphores[currentFrame].semaphore,
           inFlightVulkanFences[currentFrame].fence,
           vulkanDrawCommandBuffers.commandBuffers[imageIndex]);
    {
      const auto result{
          present(presentQueue, vulkanSwapchain, imageIndex,
                  renderFinishedVulkanSemaphores[currentFrame].semaphore)};
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
  if (arguments.size() < 7)
    return EXIT_FAILURE;
  try {
    sbash64::graphics::run(arguments[1], arguments[2], arguments[3],
                           arguments[4], arguments[5], arguments[6]);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
