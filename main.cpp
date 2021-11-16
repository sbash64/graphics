#include <sbash64/graphics/glfw-wrappers.hpp>
#include <sbash64/graphics/load-object.hpp>
#include <sbash64/graphics/stbi-wrappers.hpp>
#include <sbash64/graphics/vulkan-wrappers.hpp>

#include <vulkan/vulkan_core.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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
struct UniformBufferObject {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 projection;
};

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
  std::transform(extensionProperties.begin(), extensionProperties.end(),
                 extensionNames.begin(),
                 [](const VkExtensionProperties &properties) {
                   return static_cast<const char *>(properties.extensionName);
                 });

  if (!std::includes(extensionNames.begin(), extensionNames.end(),
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

static auto suitableDevice(const std::vector<VkPhysicalDevice> &devices,
                           VkSurfaceKHR surface) -> VkPhysicalDevice {
  for (auto *const device : devices)
    if (suitable(device, surface))
      return device;
  throw std::runtime_error("failed to find a suitable GPU!");
}

static auto vulkanDevices(VkInstance instance)
    -> std::vector<VkPhysicalDevice> {
  uint32_t deviceCount{0};
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
  return devices;
}

static auto swapChainImages(VkDevice device, VkSwapchainKHR swapChain)
    -> std::vector<VkImage> {
  uint32_t imageCount{0};
  vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
  std::vector<VkImage> swapChainImages(imageCount);
  vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
                          swapChainImages.data());
  return swapChainImages;
}

static void framebufferResizeCallback(GLFWwindow *window, int /*width*/,
                                      int /*height*/) {
  auto *const framebufferResized =
      static_cast<bool *>(glfwGetWindowUserPointer(window));
  *framebufferResized = true;
}

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

static void begin(VkCommandBuffer buffer) {
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(buffer, &beginInfo);
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

static void copy(VkDevice device, VkDeviceMemory memory, const void *source,
                 size_t size) {
  void *data = nullptr;
  vkMapMemory(device, memory, 0, size, 0, &data);
  memcpy(data, source, size);
  vkUnmapMemory(device, memory);
}

static void transitionImageLayout(VkDevice device, VkCommandPool commandPool,
                                  VkQueue graphicsQueue, VkImage image,
                                  VkImageLayout oldLayout,
                                  VkImageLayout newLayout, uint32_t mipLevels) {
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

static void copyBufferToImage(VkDevice device, VkCommandPool commandPool,
                              VkQueue graphicsQueue, VkBuffer buffer,
                              VkImage image, uint32_t width, uint32_t height) {
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

static auto bufferMemory(VkDevice device, VkPhysicalDevice physicalDevice,
                         VkBuffer buffer, VkMemoryPropertyFlags flags)
    -> vulkan_wrappers::DeviceMemory {
  VkMemoryRequirements memoryRequirements;
  vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
  vulkan_wrappers::DeviceMemory memory{device, physicalDevice, flags,
                                       memoryRequirements};
  vkBindBufferMemory(device, buffer, memory.memory, 0);
  return memory;
}

static auto imageMemory(VkDevice device, VkPhysicalDevice physicalDevice,
                        VkImage image, VkMemoryPropertyFlags flags)
    -> vulkan_wrappers::DeviceMemory {
  VkMemoryRequirements memoryRequirements;
  vkGetImageMemoryRequirements(device, image, &memoryRequirements);
  vulkan_wrappers::DeviceMemory memory{device, physicalDevice, flags,
                                       memoryRequirements};
  vkBindImageMemory(device, image, memory.memory, 0);
  return memory;
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

static void generateMipmaps(VkDevice device, VkPhysicalDevice physicalDevice,
                            VkCommandPool commandPool, VkQueue graphicsQueue,
                            VkImage image, VkFormat imageFormat,
                            int32_t texWidth, int32_t texHeight,
                            uint32_t mipLevels) {
  // Check if image format supports linear blitting
  VkFormatProperties formatProperties;
  vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat,
                                      &formatProperties);

  if (!(formatProperties.optimalTilingFeatures &
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
    throw std::runtime_error(
        "texture image format does not support linear blitting!");
  }

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

  int32_t mipWidth = texWidth;
  int32_t mipHeight = texHeight;

  for (uint32_t i = 1; i < mipLevels; i++) {
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

static auto swapChainImageViews(VkDevice device,
                                VkPhysicalDevice physicalDevice,
                                VkSurfaceKHR surface,
                                const std::vector<VkImage> &swapChainImages)
    -> std::vector<vulkan_wrappers::ImageView> {
  auto formatCount{vulkanCountFromPhysicalDevice(
      physicalDevice, [surface](VkPhysicalDevice device_, uint32_t *count) {
        vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, count, nullptr);
      })};
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                       formats.data());
  const auto format{swapSurfaceFormat(formats).format};
  std::vector<vulkan_wrappers::ImageView> swapChainImageViews;
  std::transform(swapChainImages.begin(), swapChainImages.end(),
                 std::back_inserter(swapChainImageViews),
                 [device, format](VkImage image) {
                   return vulkan_wrappers::ImageView{
                       device, image, format, VK_IMAGE_ASPECT_COLOR_BIT, 1};
                 });
  return swapChainImageViews;
}

struct VulkanDescriptor {
  vulkan_wrappers::DescriptorPool pool;
  std::vector<VkDescriptorSet> sets;
};

static auto descriptor(
    const vulkan_wrappers::Device &vulkanDevice,
    const vulkan_wrappers::ImageView &vulkanTextureImageView,
    const vulkan_wrappers::Sampler &vulkanTextureSampler,
    const vulkan_wrappers::DescriptorSetLayout &vulkanDescriptorSetLayout,
    const std::vector<VkImage> &swapChainImages,
    const std::vector<vulkan_wrappers::Buffer> &vulkanUniformBuffers)
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

  // no deallocate needed here per tutorial
  throwOnError(
      [&]() {
        return vkAllocateDescriptorSets(vulkanDevice.device, &allocInfo,
                                        descriptorSets.data());
      },
      "failed to allocate descriptor sets!");

  for (size_t i{0}; i < swapChainImages.size(); i++) {
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = vulkanUniformBuffers[i].buffer;
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

static void updateUniformBuffer(
    const vulkan_wrappers::Device &vulkanDevice,
    const vulkan_wrappers::DeviceMemory &vulkanUniformBuffersMemory,
    VkExtent2D swapChainExtent, int rotationAngleCentidegrees) {
  UniformBufferObject ubo{};
  ubo.model = glm::rotate(
      glm::mat4(1.0F),
      glm::radians(static_cast<float>(rotationAngleCentidegrees) / 100),
      glm::vec3(0.0F, 1.0F, 0.0F));
  ubo.view =
      glm::lookAt(glm::vec3(8.0F, 8.0F, 8.0F), glm::vec3(4.0F, 5.0F, 4.0F),
                  glm::vec3(0.0F, 1.0F, 1.0F));
  ubo.projection =
      glm::perspective(glm::radians(45.0F),
                       static_cast<float>(swapChainExtent.width) /
                           static_cast<float>(swapChainExtent.height),
                       0.1F, 20.0F);
  ubo.projection[1][1] *= -1;
  copy(vulkanDevice.device, vulkanUniformBuffersMemory.memory, &ubo,
       sizeof(ubo));
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

struct VulkanBufferWithMemory {
  vulkan_wrappers::Buffer buffer;
  vulkan_wrappers::DeviceMemory memory;
};

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

static void run(const std::string &vertexShaderCodePath,
                const std::string &fragmentShaderCodePath,
                const std::string &objectPath,
                const std::vector<std::string> &textureImagePaths) {
  const glfw_wrappers::Init glfwInitialization;

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  const glfw_wrappers::Window glfwWindow{1280, 720};
  auto framebufferResized{false};
  glfwSetWindowUserPointer(glfwWindow.window, &framebufferResized);
  glfwSetFramebufferSizeCallback(glfwWindow.window, framebufferResizeCallback);

  const vulkan_wrappers::Instance vulkanInstance;
  const vulkan_wrappers::Surface vulkanSurface{vulkanInstance.instance,
                                               glfwWindow.window};
  auto *vulkanPhysicalDevice{suitableDevice(
      vulkanDevices(vulkanInstance.instance), vulkanSurface.surface)};
  const vulkan_wrappers::Device vulkanDevice{vulkanPhysicalDevice,
                                             vulkanSurface.surface};
  {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(vulkanPhysicalDevice, &deviceProperties);
    std::cout << "selected physical device \"" << deviceProperties.deviceName
              << "\"\n";
  }

  VkQueue graphicsQueue{nullptr};
  vkGetDeviceQueue(vulkanDevice.device,
                   graphicsSupportingQueueFamilyIndex(vulkanPhysicalDevice), 0,
                   &graphicsQueue);
  VkQueue presentQueue{nullptr};
  vkGetDeviceQueue(vulkanDevice.device,
                   presentSupportingQueueFamilyIndex(vulkanPhysicalDevice,
                                                     vulkanSurface.surface),
                   0, &presentQueue);

  const vulkan_wrappers::CommandPool vulkanCommandPool{vulkanDevice.device,
                                                       vulkanPhysicalDevice};

  std::vector<VulkanImage> textureImages;
  std::transform(textureImagePaths.begin(), textureImagePaths.end(),
                 std::back_inserter(textureImages),
                 [&vulkanDevice, vulkanPhysicalDevice, &vulkanCommandPool,
                  graphicsQueue](const std::string &path) {
                   return textureImage(
                       vulkanDevice.device, vulkanPhysicalDevice,
                       vulkanCommandPool.commandPool, graphicsQueue, path);
                 });

  const vulkan_wrappers::Sampler vulkanTextureSampler{vulkanDevice.device,
                                                      vulkanPhysicalDevice, 13};

  const auto modelObjects{readObjects(objectPath)};
  std::vector<VulkanDrawable> vulkanDrawables;
  std::transform(modelObjects.begin(), modelObjects.end(),
                 std::back_inserter(vulkanDrawables),
                 [&vulkanDevice, vulkanPhysicalDevice, &vulkanCommandPool,
                  graphicsQueue](const Object &object) {
                   return drawable(vulkanDevice.device, vulkanPhysicalDevice,
                                   vulkanCommandPool.commandPool, graphicsQueue,
                                   object);
                 });

  const auto maxFramesInFlight{2};

  std::vector<vulkan_wrappers::Semaphore> vulkanImageAvailableSemaphores;
  std::vector<vulkan_wrappers::Semaphore> vulkanRenderFinishedSemaphores;
  std::vector<vulkan_wrappers::Fence> vulkanInFlightFences;
  for (auto i{0}; i < maxFramesInFlight; ++i) {
    vulkanImageAvailableSemaphores.emplace_back(vulkanDevice.device);
    vulkanRenderFinishedSemaphores.emplace_back(vulkanDevice.device);
    vulkanInFlightFences.emplace_back(vulkanDevice.device);
  }
  std::vector<VkFence> imagesInFlight;

  const vulkan_wrappers::DescriptorSetLayout vulkanDescriptorSetLayout{
      vulkanDevice.device};

  const vulkan_wrappers::Swapchain vulkanSwapchain{
      vulkanDevice.device, vulkanPhysicalDevice, vulkanSurface.surface,
      glfwWindow.window};
  const auto swapChainImages{graphics::swapChainImages(
      vulkanDevice.device, vulkanSwapchain.swapChain)};
  const auto swapChainImageViews{
      graphics::swapChainImageViews(vulkanDevice.device, vulkanPhysicalDevice,
                                    vulkanSurface.surface, swapChainImages)};

  const vulkan_wrappers::RenderPass vulkanRenderPass{
      vulkanDevice.device, vulkanPhysicalDevice, vulkanSurface.surface};

  const vulkan_wrappers::PipelineLayout vulkanPipelineLayout{
      vulkanDevice.device, vulkanDescriptorSetLayout.descriptorSetLayout};

  const vulkan_wrappers::Pipeline vulkanPipeline{
      vulkanDevice.device,         vulkanPhysicalDevice,
      vulkanSurface.surface,       vulkanPipelineLayout.pipelineLayout,
      vulkanRenderPass.renderPass, vertexShaderCodePath,
      fragmentShaderCodePath,      glfwWindow.window};

  const auto swapChainExtent{swapExtent(
      vulkanPhysicalDevice, vulkanSurface.surface, glfwWindow.window)};

  auto formatCount{vulkanCountFromPhysicalDevice(
      vulkanPhysicalDevice,
      [&vulkanSurface](VkPhysicalDevice device_, uint32_t *count) {
        vkGetPhysicalDeviceSurfaceFormatsKHR(device_, vulkanSurface.surface,
                                             count, nullptr);
      })};
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(vulkanPhysicalDevice,
                                       vulkanSurface.surface, &formatCount,
                                       formats.data());
  const auto colorFormat{swapSurfaceFormat(formats).format};
  const vulkan_wrappers::Image vulkanColorImage{
      vulkanDevice.device,
      swapChainExtent.width,
      swapChainExtent.height,
      colorFormat,
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
          VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
      1,
      getMaxUsableSampleCount(vulkanPhysicalDevice)};
  const auto vulkanColorImageMemory{
      imageMemory(vulkanDevice.device, vulkanPhysicalDevice,
                  vulkanColorImage.image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
  const vulkan_wrappers::ImageView vulkanColorImageView{
      vulkanDevice.device, vulkanColorImage.image, colorFormat,
      VK_IMAGE_ASPECT_COLOR_BIT, 1};

  const auto depthFormat{findDepthFormat(vulkanPhysicalDevice)};
  const vulkan_wrappers::Image vulkanDepthImage{
      vulkanDevice.device,
      swapChainExtent.width,
      swapChainExtent.height,
      depthFormat,
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      1,
      getMaxUsableSampleCount(vulkanPhysicalDevice)};
  const auto vulkanDepthImageMemory{
      imageMemory(vulkanDevice.device, vulkanPhysicalDevice,
                  vulkanDepthImage.image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
  const vulkan_wrappers::ImageView vulkanDepthImageView{
      vulkanDevice.device, vulkanDepthImage.image, depthFormat,
      VK_IMAGE_ASPECT_DEPTH_BIT, 1};

  std::vector<vulkan_wrappers::Framebuffer> vulkanFrameBuffers;
  std::transform(swapChainImageViews.begin(), swapChainImageViews.end(),
                 std::back_inserter(vulkanFrameBuffers),
                 [&vulkanDevice, vulkanPhysicalDevice, &vulkanSurface,
                  &vulkanRenderPass, &vulkanColorImageView,
                  &vulkanDepthImageView,
                  &glfwWindow](const vulkan_wrappers::ImageView &imageView) {
                   return vulkan_wrappers::Framebuffer{
                       vulkanDevice.device,
                       vulkanPhysicalDevice,
                       vulkanSurface.surface,
                       vulkanRenderPass.renderPass,
                       {vulkanColorImageView.view, vulkanDepthImageView.view,
                        imageView.view},
                       glfwWindow.window};
                 });

  std::vector<vulkan_wrappers::Buffer> vulkanUniformBuffers;
  std::vector<vulkan_wrappers::DeviceMemory> vulkanUniformBuffersMemory;

  for (size_t i{0}; i < swapChainImages.size(); i++) {
    VkDeviceSize bufferSize{sizeof(UniformBufferObject)};
    vulkanUniformBuffers.emplace_back(
        vulkanDevice.device, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, bufferSize);
    vulkanUniformBuffersMemory.push_back(
        bufferMemory(vulkanDevice.device, vulkanPhysicalDevice,
                     vulkanUniformBuffers.back().buffer,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
  }

  std::vector<VulkanDescriptor> vulkanTextureImageDescriptors;
  std::transform(
      textureImages.begin(), textureImages.end(),
      std::back_inserter(vulkanTextureImageDescriptors),
      [&vulkanDevice, &vulkanTextureSampler, &vulkanDescriptorSetLayout,
       &swapChainImages, &vulkanUniformBuffers](const VulkanImage &image) {
        return graphics::descriptor(
            vulkanDevice, image.view, vulkanTextureSampler,
            vulkanDescriptorSetLayout, swapChainImages, vulkanUniformBuffers);
      });

  const vulkan_wrappers::CommandBuffers vulkanCommandBuffers{
      vulkanDevice.device, vulkanCommandPool.commandPool,
      vulkanFrameBuffers.size()};

  for (auto i{0U}; i < vulkanCommandBuffers.commandBuffers.size(); i++) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    throwOnError(
        [&]() {
          return vkBeginCommandBuffer(vulkanCommandBuffers.commandBuffers[i],
                                      &beginInfo);
        },
        "failed to begin recording command buffer!");

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = vulkanRenderPass.renderPass;
    renderPassInfo.framebuffer = vulkanFrameBuffers.at(i).framebuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapExtent(
        vulkanPhysicalDevice, vulkanSurface.surface, glfwWindow.window);

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.0F, 0.0F, 0.0F, 1.0F}};
    clearValues[1].depthStencil = {1.0F, 0};
    renderPassInfo.clearValueCount = clearValues.size();
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(vulkanCommandBuffers.commandBuffers[i],
                         &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(vulkanCommandBuffers.commandBuffers[i],
                      VK_PIPELINE_BIND_POINT_GRAPHICS, vulkanPipeline.pipeline);
    std::vector<unsigned> objectToTextureMapping{7, 5, 2, 3, 1, 6, 4, 0};
    for (auto j{0U}; j < vulkanDrawables.size(); ++j) {
      std::array<VkBuffer, 1> vertexBuffers = {
          vulkanDrawables.at(j).vertexBufferWithMemory.buffer.buffer};
      std::array<VkDeviceSize, 1> offsets = {0};
      vkCmdBindVertexBuffers(vulkanCommandBuffers.commandBuffers[i], 0, 1,
                             vertexBuffers.data(), offsets.data());
      vkCmdBindIndexBuffer(
          vulkanCommandBuffers.commandBuffers[i],
          vulkanDrawables.at(j).indexBufferWithMemory.buffer.buffer, 0,
          VK_INDEX_TYPE_UINT32);
      vkCmdBindDescriptorSets(
          vulkanCommandBuffers.commandBuffers[i],
          VK_PIPELINE_BIND_POINT_GRAPHICS, vulkanPipelineLayout.pipelineLayout,
          0, 1,
          &vulkanTextureImageDescriptors.at(objectToTextureMapping.at(j))
               .sets[i],
          0, nullptr);
      vkCmdDrawIndexed(vulkanCommandBuffers.commandBuffers[i],
                       static_cast<uint32_t>(modelObjects.at(j).indices.size()),
                       1, 0, 0, 0);
    }

    vkCmdEndRenderPass(vulkanCommandBuffers.commandBuffers[i]);
    throwOnError(
        [&]() {
          return vkEndCommandBuffer(vulkanCommandBuffers.commandBuffers[i]);
        },
        "failed to record command buffer!");
  }

  imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
  auto recreatingSwapChain{false};
  auto rotationAngleCentidegrees{0};
  auto currentFrame{0U};
  while (!recreatingSwapChain) {
    if (glfwWindowShouldClose(glfwWindow.window) != 0) {
      break;
    }

    glfwPollEvents();
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

    updateUniformBuffer(vulkanDevice, vulkanUniformBuffersMemory[imageIndex],
                        swapChainExtent, rotationAngleCentidegrees);

    if ((rotationAngleCentidegrees += 9) == 36000)
      rotationAngleCentidegrees = 0;

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
          framebufferResized) {
        framebufferResized = false;
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
    std::vector<std::string> texturePaths;
    for (const auto &entry : std::filesystem::directory_iterator{arguments[4]})
      texturePaths.push_back(entry.path());
    sbash64::graphics::run(arguments[1], arguments[2], arguments[3],
                           texturePaths);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
