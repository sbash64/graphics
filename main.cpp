#include <sbash64/graphics/vulkan-wrappers.hpp>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace glfw_wrappers {
struct Init {
  Init() { glfwInit(); }

  ~Init() { glfwTerminate(); }

  Init(const Init &) = delete;
  auto operator=(const Init &) -> Init & = delete;
  Init(Init &&) = delete;
  auto operator=(Init &&) -> Init & = delete;
};

struct Window {
  Window(int width, int height)
      : window{glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr)} {}

  ~Window() { glfwDestroyWindow(window); }

  Window(const Window &) = delete;
  auto operator=(const Window &) -> Window & = delete;
  Window(Window &&) = delete;
  auto operator=(Window &&) -> Window & = delete;

  GLFWwindow *window;
};
} // namespace glfw_wrappers

struct UniformBufferObject {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

constexpr auto windowWidth{800};
constexpr auto windowHeight{600};

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
                   return properties.extensionName;
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
  for (const auto &device : devices)
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

static void framebufferResizeCallback(GLFWwindow *window, int width,
                                      int height) {
  auto *framebufferResized =
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

static void copyBuffer(VkDevice device, VkCommandPool commandPool,
                       VkQueue graphicsQueue, VkBuffer sourceBuffer,
                       VkBuffer destinationBuffer, VkDeviceSize size) {
  vulkan_wrappers::CommandBuffers vulkanCommandBuffers{device, commandPool, 1};

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(vulkanCommandBuffers.commandBuffers.at(0), &beginInfo);

  std::array<VkBufferCopy, 1> copyRegion{};
  copyRegion.at(0).size = size;
  vkCmdCopyBuffer(vulkanCommandBuffers.commandBuffers.at(0), sourceBuffer,
                  destinationBuffer, copyRegion.size(), copyRegion.data());

  vkEndCommandBuffer(vulkanCommandBuffers.commandBuffers.at(0));

  std::array<VkSubmitInfo, 1> submitInfo{};
  submitInfo.at(0).sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.at(0).commandBufferCount =
      static_cast<uint32_t>(vulkanCommandBuffers.commandBuffers.size());
  submitInfo.at(0).pCommandBuffers = vulkanCommandBuffers.commandBuffers.data();

  vkQueueSubmit(graphicsQueue, submitInfo.size(), submitInfo.data(),
                VK_NULL_HANDLE);
  vkQueueWaitIdle(graphicsQueue);
}

static void run(const std::string &vertexShaderCodePath,
                const std::string &fragmentShaderCodePath) {
  const glfw_wrappers::Init glfwInitialization;

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  const glfw_wrappers::Window glfwWindow{windowWidth, windowHeight};
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

  const std::vector<Vertex> vertices = {{{-0.5F, -0.5F}, {1.0F, 0.0F, 0.0F}},
                                        {{0.5F, -0.5F}, {0.0F, 1.0F, 0.0F}},
                                        {{0.5F, 0.5F}, {0.0F, 0.0F, 1.0F}},
                                        {{-0.5F, 0.5F}, {1.0F, 1.0F, 1.0F}}};

  const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

  VkDeviceSize vertexBufferSize{sizeof(vertices[0]) * vertices.size()};

  const vulkan_wrappers::Buffer vulkanVertexBuffer{
      vulkanDevice.device,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      vertexBufferSize};
  const vulkan_wrappers::DeviceMemory vulkanVertexBufferMemory{
      vulkanDevice.device, vulkanPhysicalDevice, vulkanVertexBuffer.buffer,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
  vkBindBufferMemory(vulkanDevice.device, vulkanVertexBuffer.buffer,
                     vulkanVertexBufferMemory.memory, 0);
  {
    const vulkan_wrappers::Buffer vulkanStagingBuffer{
        vulkanDevice.device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        vertexBufferSize};
    const vulkan_wrappers::DeviceMemory vulkanStagingBufferMemory{
        vulkanDevice.device, vulkanPhysicalDevice, vulkanStagingBuffer.buffer,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    vkBindBufferMemory(vulkanDevice.device, vulkanStagingBuffer.buffer,
                       vulkanStagingBufferMemory.memory, 0);

    void *data = nullptr;
    vkMapMemory(vulkanDevice.device, vulkanStagingBufferMemory.memory, 0,
                vertexBufferSize, 0, &data);
    memcpy(data, vertices.data(), vertexBufferSize);
    vkUnmapMemory(vulkanDevice.device, vulkanStagingBufferMemory.memory);

    copyBuffer(vulkanDevice.device, vulkanCommandPool.commandPool,
               graphicsQueue, vulkanStagingBuffer.buffer,
               vulkanVertexBuffer.buffer, vertexBufferSize);
  }

  VkDeviceSize indexBufferSize{sizeof(indices[0]) * indices.size()};

  const vulkan_wrappers::Buffer vulkanIndexBuffer{
      vulkanDevice.device,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      indexBufferSize};
  const vulkan_wrappers::DeviceMemory vulkanIndexBufferMemory{
      vulkanDevice.device, vulkanPhysicalDevice, vulkanIndexBuffer.buffer,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
  vkBindBufferMemory(vulkanDevice.device, vulkanIndexBuffer.buffer,
                     vulkanIndexBufferMemory.memory, 0);
  {
    const vulkan_wrappers::Buffer vulkanStagingBuffer{
        vulkanDevice.device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, indexBufferSize};
    const vulkan_wrappers::DeviceMemory vulkanStagingBufferMemory{
        vulkanDevice.device, vulkanPhysicalDevice, vulkanStagingBuffer.buffer,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    vkBindBufferMemory(vulkanDevice.device, vulkanStagingBuffer.buffer,
                       vulkanStagingBufferMemory.memory, 0);

    void *data = nullptr;
    vkMapMemory(vulkanDevice.device, vulkanStagingBufferMemory.memory, 0,
                indexBufferSize, 0, &data);
    memcpy(data, indices.data(), indexBufferSize);
    vkUnmapMemory(vulkanDevice.device, vulkanStagingBufferMemory.memory);

    copyBuffer(vulkanDevice.device, vulkanCommandPool.commandPool,
               graphicsQueue, vulkanStagingBuffer.buffer,
               vulkanIndexBuffer.buffer, indexBufferSize);
  }

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

  auto currentFrame{0};
  auto playing{true};
  while (playing) {
    const vulkan_wrappers::Swapchain vulkanSwapchain{
        vulkanDevice.device, vulkanPhysicalDevice, vulkanSurface.surface,
        glfwWindow.window};

    const auto swapChainImages{
        ::swapChainImages(vulkanDevice.device, vulkanSwapchain.swapChain)};
    std::vector<vulkan_wrappers::ImageView> swapChainImageViews;
    std::transform(
        swapChainImages.begin(), swapChainImages.end(),
        std::back_inserter(swapChainImageViews),
        [&vulkanDevice, vulkanPhysicalDevice, &vulkanSurface](VkImage image) {
          return vulkan_wrappers::ImageView{vulkanDevice.device,
                                            vulkanPhysicalDevice,
                                            vulkanSurface.surface, image};
        });

    const vulkan_wrappers::RenderPass vulkanRenderPass{
        vulkanDevice.device, vulkanPhysicalDevice, vulkanSurface.surface};

    const vulkan_wrappers::PipelineLayout vulkanPipelineLayout{
        vulkanDevice.device, vulkanDescriptorSetLayout.descriptorSetLayout};

    const vulkan_wrappers::Pipeline vulkanPipeline{
        vulkanDevice.device,         vulkanPhysicalDevice,
        vulkanSurface.surface,       vulkanPipelineLayout.pipelineLayout,
        vulkanRenderPass.renderPass, vertexShaderCodePath,
        fragmentShaderCodePath,      glfwWindow.window};

    std::vector<vulkan_wrappers::Framebuffer> vulkanFrameBuffers;
    std::transform(swapChainImageViews.begin(), swapChainImageViews.end(),
                   std::back_inserter(vulkanFrameBuffers),
                   [&vulkanDevice, vulkanPhysicalDevice, &vulkanSurface,
                    &vulkanRenderPass,
                    &glfwWindow](const vulkan_wrappers::ImageView &imageView) {
                     return vulkan_wrappers::Framebuffer{
                         vulkanDevice.device,   vulkanPhysicalDevice,
                         vulkanSurface.surface, vulkanRenderPass.renderPass,
                         imageView.view,        glfwWindow.window};
                   });

    std::vector<vulkan_wrappers::Buffer> vulkanUniformBuffers;
    std::vector<vulkan_wrappers::DeviceMemory> vulkanUniformBuffersMemory;

    for (size_t i = 0; i < swapChainImages.size(); i++) {
      VkDeviceSize bufferSize = sizeof(UniformBufferObject);
      vulkanUniformBuffers.emplace_back(
          vulkanDevice.device, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, bufferSize);
      vulkanUniformBuffersMemory.emplace_back(
          vulkanDevice.device, vulkanPhysicalDevice,
          vulkanUniformBuffers.back().buffer,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      vkBindBufferMemory(vulkanDevice.device,
                         vulkanUniformBuffers.back().buffer,
                         vulkanUniformBuffersMemory.back().memory, 0);
    }

    const vulkan_wrappers::DescriptorPool vulkanDescriptorPool{
        vulkanDevice.device, swapChainImages};
    std::vector<VkDescriptorSet> descriptorSets(swapChainImages.size());
    {
      std::vector<VkDescriptorSetLayout> layouts(
          swapChainImages.size(),
          vulkanDescriptorSetLayout.descriptorSetLayout);
      VkDescriptorSetAllocateInfo allocInfo{};
      allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      allocInfo.descriptorPool = vulkanDescriptorPool.descriptorPool;
      allocInfo.descriptorSetCount =
          static_cast<uint32_t>(swapChainImages.size());
      allocInfo.pSetLayouts = layouts.data();

      if (vkAllocateDescriptorSets(vulkanDevice.device, &allocInfo,
                                   descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
      }

      for (size_t i = 0; i < swapChainImages.size(); i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = vulkanUniformBuffers[i].buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSets[i];
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(vulkanDevice.device, 1, &descriptorWrite, 0,
                               nullptr);
      }
    }

    const vulkan_wrappers::CommandBuffers vulkanCommandBuffers{
        vulkanDevice.device, vulkanCommandPool.commandPool,
        vulkanFrameBuffers.size()};

    for (auto i{0}; i < vulkanCommandBuffers.commandBuffers.size(); i++) {
      VkCommandBufferBeginInfo beginInfo{};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      if (vkBeginCommandBuffer(vulkanCommandBuffers.commandBuffers[i],
                               &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording command buffer!");

      VkRenderPassBeginInfo renderPassInfo{};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      renderPassInfo.renderPass = vulkanRenderPass.renderPass;
      renderPassInfo.framebuffer = vulkanFrameBuffers.at(i).framebuffer;
      renderPassInfo.renderArea.offset = {0, 0};

      renderPassInfo.renderArea.extent = swapExtent(
          vulkanPhysicalDevice, vulkanSurface.surface, glfwWindow.window);

      const std::array<VkClearValue, 1> clearColor = {
          {{{{0.0F, 0.0F, 0.0F, 1.0F}}}}};
      renderPassInfo.clearValueCount = clearColor.size();
      renderPassInfo.pClearValues = clearColor.data();

      vkCmdBeginRenderPass(vulkanCommandBuffers.commandBuffers[i],
                           &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

      vkCmdBindPipeline(vulkanCommandBuffers.commandBuffers[i],
                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                        vulkanPipeline.pipeline);

      std::array<VkBuffer, 1> vertexBuffers = {vulkanVertexBuffer.buffer};
      std::array<VkDeviceSize, 1> offsets = {0};
      vkCmdBindVertexBuffers(vulkanCommandBuffers.commandBuffers[i], 0, 1,
                             vertexBuffers.data(), offsets.data());

      vkCmdBindIndexBuffer(vulkanCommandBuffers.commandBuffers[i],
                           vulkanIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

      vkCmdBindDescriptorSets(vulkanCommandBuffers.commandBuffers[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              vulkanPipelineLayout.pipelineLayout, 0, 1,
                              &descriptorSets[i], 0, nullptr);

      vkCmdDrawIndexed(vulkanCommandBuffers.commandBuffers[i],
                       static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

      vkCmdEndRenderPass(vulkanCommandBuffers.commandBuffers[i]);

      if (vkEndCommandBuffer(vulkanCommandBuffers.commandBuffers[i]) !=
          VK_SUCCESS)
        throw std::runtime_error("failed to record command buffer!");
    }

    const auto swapChainExtent{swapExtent(
        vulkanPhysicalDevice, vulkanSurface.surface, glfwWindow.window)};

    imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
    auto recreatingSwapChain{false};
    while (!recreatingSwapChain) {
      if (glfwWindowShouldClose(glfwWindow.window) != 0) {
        playing = false;
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
      {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(
                         currentTime - startTime)
                         .count();

        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f),
                                glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f),
                               glm::vec3(0.0f, 0.0f, 0.0f),
                               glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(
            glm::radians(45.0f),
            swapChainExtent.width / static_cast<float>(swapChainExtent.height),
            0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        void *data = nullptr;
        vkMapMemory(vulkanDevice.device,
                    vulkanUniformBuffersMemory[imageIndex].memory, 0,
                    sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(vulkanDevice.device,
                      vulkanUniformBuffersMemory[imageIndex].memory);
      }

      if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        vkWaitForFences(vulkanDevice.device, 1, &imagesInFlight[imageIndex],
                        VK_TRUE, UINT64_MAX);

      imagesInFlight[imageIndex] = vulkanInFlightFences[currentFrame].fence;

      VkSubmitInfo submitInfo{};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

      const std::array<VkSemaphore, 1> waitSemaphores = {
          vulkanImageAvailableSemaphores[currentFrame].semaphore};
      const std::array<VkPipelineStageFlags, 1> waitStages = {
          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
      submitInfo.waitSemaphoreCount = waitSemaphores.size();
      submitInfo.pWaitSemaphores = waitSemaphores.data();
      submitInfo.pWaitDstStageMask = waitStages.data();

      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers =
          &vulkanCommandBuffers.commandBuffers[imageIndex];

      const std::array<VkSemaphore, 1> signalSemaphores = {
          vulkanRenderFinishedSemaphores[currentFrame].semaphore};
      submitInfo.signalSemaphoreCount = signalSemaphores.size();
      submitInfo.pSignalSemaphores = signalSemaphores.data();

      vkResetFences(vulkanDevice.device, 1,
                    &vulkanInFlightFences[currentFrame].fence);

      if (vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                        vulkanInFlightFences[currentFrame].fence) != VK_SUCCESS)
        throw std::runtime_error("failed to submit draw command buffer!");

      VkPresentInfoKHR presentInfo{};
      presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

      presentInfo.waitSemaphoreCount = waitSemaphores.size();
      presentInfo.pWaitSemaphores = signalSemaphores.data();

      const std::array<VkSwapchainKHR, 1> swapChains = {
          vulkanSwapchain.swapChain};
      presentInfo.swapchainCount = swapChains.size();
      presentInfo.pSwapchains = swapChains.data();

      presentInfo.pImageIndices = &imageIndex;

      {
        const auto result{vkQueuePresentKHR(presentQueue, &presentInfo)};

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
  }

  vkDeviceWaitIdle(vulkanDevice.device);
}

int main(int argc, char *argv[]) {
  const std::span<char *> arguments{
      argv, static_cast<std::span<char *>::size_type>(argc)};
  if (arguments.size() < 3)
    return EXIT_FAILURE;
  try {
    run(arguments[1], arguments[2]);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
