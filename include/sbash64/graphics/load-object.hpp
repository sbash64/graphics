#ifndef SBASH64_GRAPHICS_LOAD_OBJECT_HPP_
#define SBASH64_GRAPHICS_LOAD_OBJECT_HPP_

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <vector>

namespace sbash64::graphics {
struct Vertex {
  glm::vec3 pos;
  glm::vec2 texCoord;

  auto operator==(const Vertex &) const -> bool = default;
};
} // namespace sbash64::graphics

namespace std {
template <> struct hash<sbash64::graphics::Vertex> {
  auto operator()(sbash64::graphics::Vertex const &vertex) const -> size_t {
    return (hash<glm::vec3>()(vertex.pos) >> 1U) ^
           (hash<glm::vec2>()(vertex.texCoord) << 1U);
  }
};
} // namespace std

namespace sbash64::graphics {
struct IndexedVertices {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
};

auto readIndexedVertices(const std::string &path) -> IndexedVertices;
} // namespace sbash64::graphics

#endif
