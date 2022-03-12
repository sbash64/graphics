#include <sbash64/graphics/load-scene.hpp>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include <span>

namespace sbash64::graphics {
template <typename T>
auto span(const tinygltf::Model &model, int index) -> std::span<const T> {
  const auto &accessor = model.accessors[index];
  const auto &bufferView = model.bufferViews[accessor.bufferView];
  const auto &buffer = model.buffers[bufferView.buffer];
  return {reinterpret_cast<const T *>(
              &buffer.data[accessor.byteOffset + bufferView.byteOffset]),
          accessor.count};
}

template <typename T>
void appendIndices(std::vector<uint32_t> &indexBuffer, uint32_t vertexStart,
                   const tinygltf::Model &model,
                   const tinygltf::Primitive &primitive) {
  for (const auto index : span<T>(model, primitive.indices))
    indexBuffer.push_back(index + vertexStart);
}

template <typename T>
auto span(const tinygltf::Model &model, const tinygltf::Primitive &primitive,
          const std::string &attribute) -> std::span<const T> {
  if (primitive.attributes.contains(attribute))
    return graphics::span<T>(model, primitive.attributes.at(attribute));
  return {};
}

static void loadNode(std::vector<std::unique_ptr<Node>> &nodes,
                     std::vector<uint32_t> &indexBuffer,
                     std::vector<AnimatingVertex> &vertexBuffer,
                     const tinygltf::Model &gltfModel, Node *parent,
                     int nodeIndex) {
  const auto gltfNode{gltfModel.nodes.at(nodeIndex)};
  auto node{std::make_unique<Node>()};
  node->parent = parent;
  node->matrix = glm::mat4(1.0F);
  node->index = nodeIndex;
  node->skin = gltfNode.skin;

  // Get the local node matrix
  // It's either made up from translation, rotation, scale or a 4x4 matrix
  if (gltfNode.translation.size() == 3)
    node->translation = glm::make_vec3(gltfNode.translation.data());
  if (gltfNode.rotation.size() == 4) {
    glm::quat q = glm::make_quat(gltfNode.rotation.data());
    node->rotation = glm::mat4(q);
  }
  if (gltfNode.scale.size() == 3)
    node->scale = glm::make_vec3(gltfNode.scale.data());
  if (gltfNode.matrix.size() == 16)
    node->matrix = glm::make_mat4x4(gltfNode.matrix.data());

  for (const auto child : gltfNode.children)
    loadNode(nodes, indexBuffer, vertexBuffer, gltfModel, node.get(), child);

  // If the node contains mesh data, we load vertices and indices from the
  // buffers In glTF this is done via accessors and buffer views
  if (gltfNode.mesh > -1) {
    for (const auto &gltfPrimitive :
         gltfModel.meshes[gltfNode.mesh].primitives) {
      Primitive primitive{};
      primitive.materialIndex = gltfPrimitive.material;
      const auto vertexStart = static_cast<uint32_t>(vertexBuffer.size());
      // Vertices
      {
        const auto positionBuffer{
            span<float>(gltfModel, gltfPrimitive, "POSITION")};
        const auto normalsBuffer{
            span<float>(gltfModel, gltfPrimitive, "NORMAL")};
        const auto texCoordsBuffer{
            span<float>(gltfModel, gltfPrimitive, "TEXCOORD_0")};
        const auto jointIndicesBuffer{
            span<uint16_t>(gltfModel, gltfPrimitive, "JOINTS_0")};
        const auto jointWeightsBuffer{
            span<float>(gltfModel, gltfPrimitive, "WEIGHTS_0")};

        // Append data to model's vertex buffer
        for (size_t v = 0; v < positionBuffer.size(); v++) {
          AnimatingVertex vertex{};
          vertex.pos = glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0F);
          vertex.normal = glm::normalize(glm::vec3(
              !normalsBuffer.empty() ? glm::make_vec3(&normalsBuffer[v * 3])
                                     : glm::vec3(0.0F)));
          vertex.uv = !texCoordsBuffer.empty()
                          ? glm::make_vec2(&texCoordsBuffer[v * 2])
                          : glm::vec3(0.0F);
          vertex.color = glm::vec3(1.0F);
          const auto hasSkin{!jointIndicesBuffer.empty() &&
                             !jointWeightsBuffer.empty()};
          vertex.jointIndices =
              hasSkin ? glm::vec4(glm::make_vec4(&jointIndicesBuffer[v * 4]))
                      : glm::vec4(0.0F);
          vertex.jointWeights = hasSkin
                                    ? glm::make_vec4(&jointWeightsBuffer[v * 4])
                                    : glm::vec4(0.0F);
          vertexBuffer.push_back(vertex);
        }
      }
      // Indices
      primitive.firstIndex = static_cast<uint32_t>(indexBuffer.size());
      {
        const auto &accessor = gltfModel.accessors[gltfPrimitive.indices];
        primitive.indexCount = static_cast<uint32_t>(accessor.count);
        switch (accessor.componentType) {
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
          appendIndices<uint32_t>(indexBuffer, vertexStart, gltfModel,
                                  gltfPrimitive);
          break;
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
          appendIndices<uint16_t>(indexBuffer, vertexStart, gltfModel,
                                  gltfPrimitive);
          break;
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
          appendIndices<uint8_t>(indexBuffer, vertexStart, gltfModel,
                                 gltfPrimitive);
          break;
        default:
          throw std::runtime_error{"component type not supported"};
        }
      }
      node->mesh.primitives.push_back(primitive);
    }
  }

  if (parent != nullptr) {
    parent->children.push_back(std::move(node));
  } else {
    nodes.push_back(std::move(node));
  }
}

static auto findNode(Node *parent, uint32_t index) -> Node * {
  if (parent->index == index)
    return parent;
  for (auto &child : parent->children) {
    Node *node = findNode(child.get(), index);
    if (node != nullptr) {
      return node;
    }
  }
  return nullptr;
}

static auto nodeFromIndex(const std::vector<std::unique_ptr<Node>> &nodes,
                          uint32_t index) -> Node * {
  for (const auto &node : nodes) {
    auto *maybeFound = findNode(node.get(), index);
    if (maybeFound != nullptr)
      return maybeFound;
  }
  return nullptr;
}

static auto images(const tinygltf::Model &gltfModel) -> std::vector<Image> {
  std::vector<Image> images;
  transform(
      gltfModel.images.begin(), gltfModel.images.end(), back_inserter(images),
      [](const tinygltf::Image &gltfImage) {
        // We convert RGB-only images to RGBA, as most devices don't
        // support RGB-formats in Vulkan
        Image image;
        image.width = gltfImage.width;
        image.height = gltfImage.height;
        if (gltfImage.component == 3) {
          image.buffer.resize(static_cast<long>(gltfImage.width) *
                              gltfImage.height * 4);
          unsigned char *rgba = image.buffer.data();
          const auto *rgb = &gltfImage.image[0];
          for (size_t i = 0;
               i < static_cast<long>(gltfImage.width) * gltfImage.height; ++i) {
            memcpy(rgba, rgb, sizeof(unsigned char) * 3);
            rgba += 4;
            rgb += 3;
          }
        } else {
          std::span<const unsigned char> gltfBuffer{&gltfImage.image[0],
                                                    gltfImage.image.size()};
          image.buffer.resize(gltfBuffer.size());
          copy(gltfBuffer.begin(), gltfBuffer.end(), image.buffer.begin());
        }
        return image;
      });
  return images;
}

static auto materials(const tinygltf::Model &gltfModel)
    -> std::vector<Material> {
  std::vector<Material> materials;
  transform(
      gltfModel.materials.begin(), gltfModel.materials.end(),
      back_inserter(materials), [](const tinygltf::Material &gltfMaterial) {
        Material material;
        if (gltfMaterial.values.contains("baseColorFactor"))
          material.baseColorFactor = glm::make_vec4(
              gltfMaterial.values.at("baseColorFactor").ColorFactor().data());
        if (gltfMaterial.values.contains("baseColorTexture"))
          material.baseColorTextureIndex =
              gltfMaterial.values.at("baseColorTexture").TextureIndex();
        return material;
      });
  return materials;
}

auto loadScene(const tinygltf::Model &gltfModel) -> Scene {
  // Images can be stored inside the glTF (which is the case for the sample
  // model), so instead of directly loading them from disk, we fetch them
  // from the glTF loader and upload the buffers
  Scene scene;
  scene.images = images(gltfModel);
  scene.materials = materials(gltfModel);

  transform(gltfModel.textures.begin(), gltfModel.textures.end(),
            back_inserter(scene.textureIndices),
            [](const tinygltf::Texture &texture) { return texture.source; });

  for (const auto node : gltfModel.scenes.at(0).nodes)
    loadNode(scene.nodes, scene.indexBuffer, scene.vertexBuffer, gltfModel,
             nullptr, node);

  transform(gltfModel.skins.begin(), gltfModel.skins.end(),
            back_inserter(scene.skins),
            [&gltfModel, &scene](const tinygltf::Skin &gltfSkin) {
              Skin skin;
              skin.name = gltfSkin.name;
              skin.skeletonRoot = nodeFromIndex(scene.nodes, gltfSkin.skeleton);

              for (const auto jointIndex : gltfSkin.joints) {
                auto *node = nodeFromIndex(scene.nodes, jointIndex);
                if (node != nullptr)
                  skin.joints.push_back(node);
              }

              if (gltfSkin.inverseBindMatrices > -1) {
                const auto span{graphics::span<glm::mat4>(
                    gltfModel, gltfSkin.inverseBindMatrices)};
                skin.inverseBindMatrices = {span.begin(), span.end()};
              }
              return skin;
            });

  transform(
      gltfModel.animations.begin(), gltfModel.animations.end(),
      back_inserter(scene.animations),
      [&gltfModel, &scene](const tinygltf::Animation &gltfAnimation) {
        Animation animation;
        animation.name = gltfAnimation.name;
        transform(gltfAnimation.samplers.begin(), gltfAnimation.samplers.end(),
                  back_inserter(animation.samplers),
                  [&gltfModel,
                   &animation](const tinygltf::AnimationSampler &gltfSampler) {
                    AnimationSampler animationSampler;
                    animationSampler.interpolation = gltfSampler.interpolation;
                    for (const auto input :
                         span<float>(gltfModel, gltfSampler.input))
                      animationSampler.inputs.push_back(input);
                    // Adjust animation's start and end times
                    for (const auto input : animationSampler.inputs) {
                      if (input < animation.start)
                        animation.start = input;
                      if (input > animation.end)
                        animation.end = input;
                    }

                    // Read sampler keyframe output translate/rotate/scale
                    // values
                    switch (gltfModel.accessors[gltfSampler.output].type) {
                    case TINYGLTF_TYPE_VEC3:
                      for (const auto v :
                           span<glm::vec3>(gltfModel, gltfSampler.output))
                        animationSampler.outputsVec4.emplace_back(v, 0.0F);
                      break;
                    case TINYGLTF_TYPE_VEC4:
                      for (const auto v :
                           span<glm::vec4>(gltfModel, gltfSampler.output))
                        animationSampler.outputsVec4.push_back(v);
                      break;
                    default:
                      throw std::runtime_error{"unknown keyframe type"};
                    }
                    return animationSampler;
                  });
        transform(
            gltfAnimation.channels.begin(), gltfAnimation.channels.end(),
            back_inserter(animation.channels),
            [&scene](const tinygltf::AnimationChannel &gltfAnimationChannel) {
              AnimationChannel animationChannel;
              animationChannel.path = gltfAnimationChannel.target_path;
              animationChannel.samplerIndex = gltfAnimationChannel.sampler;
              animationChannel.node =
                  nodeFromIndex(scene.nodes, gltfAnimationChannel.target_node);
              return animationChannel;
            });
        return animation;
      });
  return scene;
}

auto readScene(const std::string &path) -> Scene {
  tinygltf::TinyGLTF gltf;
  tinygltf::Model gltfModel;
  std::string error;
  std::string warning;
  if (!gltf.LoadASCIIFromFile(&gltfModel, &error, &warning, path))
    throw std::runtime_error{error};
  return loadScene(gltfModel);
}
} // namespace sbash64::graphics