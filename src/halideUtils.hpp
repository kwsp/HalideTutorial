#pragma once

#include <Halide.h>
#include <fmt/core.h>
#include <opencv2/opencv.hpp>

namespace hl {

// A helper function to check if OpenCL, Metal or D3D12 is present on the host
// machine.

Halide::Target find_gpu_target() {
  // Start with a target suitable for the machine you're running this on.
  auto target = Halide::get_host_target();

  std::vector<Halide::Target::Feature> features_to_try;
  if (target.os == Halide::Target::Windows) {
    // Try D3D12 first; if that fails, try OpenCL.
    if (sizeof(void *) == 8) {
      // D3D12Compute support is only available on 64-bit systems at present.
      features_to_try.push_back(Halide::Target::D3D12Compute);
    }
    features_to_try.push_back(Halide::Target::OpenCL);
  } else if (target.os == Halide::Target::OSX) {
    // OS X doesn't update its OpenCL drivers, so they tend to be broken.
    // CUDA would also be a fine choice on machines with NVidia GPUs.
    features_to_try.push_back(Halide::Target::Metal);
  } else {
    features_to_try.push_back(Halide::Target::OpenCL);
  }
  // Uncomment the following lines to also try CUDA:
  // features_to_try.push_back(Target::CUDA);

  for (Halide::Target::Feature f : features_to_try) {
    Halide::Target new_target = target.with_feature(f);
    if (host_supports_target_device(new_target)) {
      return new_target;
    }
  }

  printf("Requested GPU(s) are not supported. (Do you have the proper hardware "
         "and/or driver installed?)\n");
  return target;
}

// Function to convert cv::Mat to Halide::Buffer<uint8_t>
Halide::Buffer<uint8_t> convertMatToHalide(const cv::Mat &mat) {
  // Ensure the Mat is continuous in memory
  if (!mat.isContinuous()) {
    // If not, throw an error or handle accordingly
    throw std::runtime_error("Input cv::Mat is not continuous. Please make a "
                             "continuous copy before conversion.");
  }

  // Determine the number of channels
  const int channels = mat.channels();

  // Create a Halide::Buffer that references the data in cv::Mat
  // Halide::Buffer expects data in (x, y, channel) order
  // OpenCV's Mat data is also stored in (x, y, channel) interleaved order

  if (channels == 1) {
    return Halide::Buffer<uint8_t>(mat.data, mat.cols, mat.rows);
  }
  return Halide::Buffer<uint8_t>(mat.data, // Pointer to the data
                                 mat.cols, // Width (x dimension)
                                 mat.rows, // Height (y dimension)
                                 channels  // Number of channels
  );
}

template <typename T>
cv::Mat convertHalideToMat(const Halide::Buffer<T> &buffer) {
  // Determine the type and create a corresponding OpenCV Mat
  int type{};
  if (buffer.channels() == 1) {
    type = CV_MAKETYPE(cv::DataType<T>::depth, 1);
  } else if (buffer.channels() == 3) {
    type = CV_MAKETYPE(cv::DataType<T>::depth, 3);
  } else {
    throw std::runtime_error("Unsupported number of channels");
  }

  // Create a Mat with the same dimensions as the Halide buffer
  cv::Mat mat(buffer.height(), buffer.width(), type);

  // Use memcpy to copy the data efficiently
  if (buffer.channels() == 1) {
    // For single-channel (grayscale) images
    std::memcpy(mat.data, buffer.data(), buffer.size_in_bytes());
  } else if (buffer.channels() == 3) {
    // For three-channel (RGB) images
    // Ensure that the layout is compatible, which is typically the case
    std::memcpy(mat.data, buffer.data(), buffer.size_in_bytes());
  }

  return mat;
}

// Apply bilinear interpolation to input at (float inX, float, inY), 2D version
template <int Dims>
[[nodiscard]] Halide::Expr
bilinear_interpolate(const Halide::ImageParam &input, Halide::Expr inX,
                     Halide::Expr inY, Halide::Expr c = 0) {
  static_assert(Dims == 2 || Dims == 3, "Unsupported number of dimensions");

  using Halide::cast;
  using Halide::clamp;
  using Halide::floor;
  using Halide::fract;
  using Halide::lerp;

  auto x0 = clamp(cast<int>(floor(inX)), 0, input.width() - 1);
  auto x1 = clamp(x0 + 1, 0, input.width() - 1);
  auto y0 = clamp(cast<int>(floor(inY)), 0, input.height() - 1);
  auto y1 = clamp(y0 + 1, 0, input.height() - 1);

  auto clamped = Halide::BoundaryConditions::repeat_edge(input);

  Halide::Expr topLeft;
  Halide::Expr topRight;
  Halide::Expr bottomLeft;
  Halide::Expr bottomRight;

  if constexpr (Dims == 2) {
    topLeft = clamped(x0, y0);
    topRight = clamped(x1, y0);
    bottomLeft = clamped(x0, y1);
    bottomRight = clamped(x1, y1);
  } else {
    topLeft = clamped(x0, y0, c);
    topRight = clamped(x1, y0, c);
    bottomLeft = clamped(x0, y1, c);
    bottomRight = clamped(x1, y1, c);
  }

  auto xf = fract(inX);
  auto yf = fract(inY);
  auto top = lerp(topLeft, topRight, xf);
  auto bottom = lerp(bottomLeft, bottomRight, xf);
  auto value = lerp(top, bottom, yf);

  return value;
}

} // namespace hl