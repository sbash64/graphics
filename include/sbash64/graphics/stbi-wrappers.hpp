#ifndef SBASH64_GRAPHICS_STBI_WRAPPERS_HPP_
#define SBASH64_GRAPHICS_STBI_WRAPPERS_HPP_

#include <stb_image.h>

#include <stdexcept>
#include <string>

namespace sbash64::graphics::stbi_wrappers {
struct Image {
  explicit Image(const std::string &);
  ~Image();

  Image(Image &&) = delete;
  auto operator=(Image &&) -> Image & = delete;
  Image(const Image &) = delete;
  auto operator=(const Image &) -> Image & = delete;

  int width{};
  int height{};
  int channels{};
  stbi_uc *pixels;
};
} // namespace sbash64::graphics::stbi_wrappers

#endif
