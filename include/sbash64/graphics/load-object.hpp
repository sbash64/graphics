#ifndef SBASH64_GRAPHICS_LOAD_OBJECT_HPP_
#define SBASH64_GRAPHICS_LOAD_OBJECT_HPP_

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <string>
#include <vector>

namespace sbash64::graphics {
struct Vertex {
  glm::vec3 position;
  glm::vec2 textureCoordinate;

  auto operator==(const Vertex &) const -> bool = default;
};
} // namespace sbash64::graphics

namespace std {
template <> struct hash<sbash64::graphics::Vertex> {
  auto operator()(sbash64::graphics::Vertex const &vertex) const -> size_t {
    return (hash<glm::vec3>()(vertex.position) >> 1U) ^
           (hash<glm::vec2>()(vertex.textureCoordinate) << 1U);
  }
};
} // namespace std

namespace sbash64::graphics {
struct StationaryObject {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  std::string textureFileName;
};

auto readStationaryObjects(const std::string &path)
    -> std::vector<StationaryObject>;
} // namespace sbash64::graphics

#endif
