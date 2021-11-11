#include <sbash64/graphics/load-object.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <unordered_map>

namespace sbash64::graphics {
auto readIndexedVertices(const std::string &path) -> IndexedVertices {
  std::vector<tinyobj::shape_t> shapes;
  tinyobj::attrib_t attrib;
  {
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                          path.c_str()))
      throw std::runtime_error(warn + err);
  }

  IndexedVertices indexedVertices;
  std::unordered_map<Vertex, uint32_t> uniqueVertices{};
  for (const auto &shape : shapes)
    for (const auto &index : shape.mesh.indices) {
      Vertex vertex{};
      vertex.pos = {attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]};

      vertex.texCoord = {attrib.texcoords[2 * index.texcoord_index + 0],
                         1.0F - attrib.texcoords[2 * index.texcoord_index + 1]};

      vertex.color = {1.0F, 1.0F, 1.0F};

      if (uniqueVertices.count(vertex) == 0) {
        uniqueVertices[vertex] =
            static_cast<uint32_t>(indexedVertices.vertices.size());
        indexedVertices.vertices.push_back(vertex);
      }

      indexedVertices.indices.push_back(uniqueVertices[vertex]);
    }
  return indexedVertices;
}
} // namespace sbash64::graphics