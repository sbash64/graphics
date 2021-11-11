#include <sbash64/graphics/stbi-wrappers.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace sbash64::graphics::stbi_wrappers {
Image::Image(const std::string &imagePath)
    : pixels{stbi_load(imagePath.c_str(), &width, &height, &channels,
                       STBI_rgb_alpha)} {
  if (pixels == nullptr)
    throw std::runtime_error("failed to load texture image!");
}

Image::~Image() { stbi_image_free(pixels); }
} // namespace sbash64::graphics::stbi_wrappers
