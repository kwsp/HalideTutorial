#pragma once

#include <Halide.h>
#include <fmt/core.h>
#include <opencv2/opencv.hpp>

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
