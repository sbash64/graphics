cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0048 NEW)
project(sbash64-learn-vulkan LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm
  GIT_TAG 0.9.9.8)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb
  GIT_TAG af1a5bc352164740c1cc1354942b1c6b72eacb8a)
FetchContent_MakeAvailable(stb)

FetchContent_Declare(
  tinyobjloader
  GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader
  GIT_TAG 097ad6a826336a76e42a27e6e531da7dbdf3fd01)
FetchContent_MakeAvailable(tinyobjloader)

FetchContent_Declare(
  tinygltf
  GIT_REPOSITORY https://github.com/syoyo/tinygltf
  GIT_TAG v2.5.0)
set(TINYGLTF_HEADER_ONLY
    ON
    CACHE INTERNAL "" FORCE)
set(TINYGLTF_INSTALL
    OFF
    CACHE INTERNAL "" FORCE)
FetchContent_MakeAvailable(tinygltf)

include(FindVulkan)

add_executable(
  sbash64-learn-vulkan-main main.cpp vulkan-wrappers.cpp glfw-wrappers.cpp
                            stbi-wrappers.cpp load-object.cpp)
target_link_libraries(sbash64-learn-vulkan-main glfw Vulkan::Vulkan glm
                      tinyobjloader)
target_include_directories(
  sbash64-learn-vulkan-main PRIVATE include ${stb_SOURCE_DIR}
                                    ${tinygltf_SOURCE_DIR})
target_compile_features(sbash64-learn-vulkan-main PRIVATE cxx_std_20)
target_compile_options(sbash64-learn-vulkan-main
                       PRIVATE "${SBASH64_LEARN_VULKAN_WARNINGS}")
