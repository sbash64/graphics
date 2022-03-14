#ifndef SBASH64_GRAPHICS_LOAD_SCENE_HPP_
#define SBASH64_GRAPHICS_LOAD_SCENE_HPP_

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace sbash64::graphics {
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

struct AnimatingVertex {
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
};

struct Image {
  std::vector<unsigned char> buffer;
  int width{};
  int height{};
};

struct Scene {
  std::vector<Image> images;
  std::vector<Material> materials;
  std::vector<int> textureIndices;
  std::vector<uint32_t> vertexIndices;
  std::vector<AnimatingVertex> vertices;
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<Skin> skins;
  std::vector<Animation> animations;
};

auto readScene(const std::string &path) -> Scene;
} // namespace sbash64::graphics

#endif
