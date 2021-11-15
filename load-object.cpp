#include <sbash64/graphics/load-object.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <unordered_map>

namespace sbash64::graphics {
auto readMeshes(const std::string &path) -> std::vector<Mesh> {
  tinyobj::ObjReader reader;
  tinyobj::ObjReaderConfig reader_config;

  if (!reader.ParseFromFile(path, reader_config))
    throw std::runtime_error{reader.Error()};

  std::vector<Mesh> meshes;
  std::transform(reader.GetShapes().begin(), reader.GetShapes().end(),
                 std::back_inserter(meshes),
                 [&reader](const tinyobj::shape_t &shape) {
                   Mesh mesh;
                   std::unordered_map<Vertex, uint32_t> uniqueVertices{};
                   for (const auto &index : shape.mesh.indices) {
                     Vertex vertex{};
                     const auto &attrib{reader.GetAttrib()};
                     vertex.pos = {attrib.vertices[3 * index.vertex_index + 0],
                                   attrib.vertices[3 * index.vertex_index + 1],
                                   attrib.vertices[3 * index.vertex_index + 2]};

                     vertex.texCoord = {
                         attrib.texcoords[2 * index.texcoord_index + 0],
                         1.0F - attrib.texcoords[2 * index.texcoord_index + 1]};

                     if (uniqueVertices.count(vertex) == 0) {
                       uniqueVertices[vertex] =
                           static_cast<uint32_t>(mesh.vertices.size());
                       mesh.vertices.push_back(vertex);
                     }

                     mesh.indices.push_back(uniqueVertices[vertex]);
                   }
                   return mesh;
                 });
  return meshes;
}
} // namespace sbash64::graphics
