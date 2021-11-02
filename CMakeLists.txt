cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0048 NEW)
project(sbash64-learn-vulkan LANGUAGES CXX)

include(FetchContent)

include(FindVulkan)

add_executable(sbash64-learn-vulkan-main main.cpp)
target_link_libraries(sbash64-learn-vulkan-main glfw Vulkan::Vulkan)
target_compile_features(sbash64-learn-vulkan-main PRIVATE cxx_std_20)
target_compile_options(sbash64-learn-vulkan-main
                       PRIVATE "${SBASH64_LEARN_VULKAN_WARNINGS}")
