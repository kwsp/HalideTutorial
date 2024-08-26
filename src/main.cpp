#include "halideUtils.hpp"
#include "resize.hpp"
#include "timeit.hpp"
#include "warpPolar.hpp"
#include <Halide.h>
#include <fmt/core.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void runWarp2D() {
  auto img =
      cv::imread("/Users/tnie/Downloads/stripes.jpg", cv::IMREAD_GRAYSCALE);
  // auto img =
  //     cv::imread("/Users/tnie/Downloads/USlog.jpg", cv::IMREAD_GRAYSCALE);
  const auto input = hl::convertMatToHalide(img);

  // Define the center, maximum radius, and whether it's a full circle
  const int r = std::min(input.width(), input.height());
  const int outX = r;
  const int outY = r;

  float centerX = static_cast<float>(r) / 2.0F;
  float centerY = static_cast<float>(r) / 2.0F;
  float maxRadius = std::min(centerX, centerY);

  Halide::ImageParam param(Halide::UInt(8), 2);

  Halide::Buffer<uint8_t> output(outX, outY);
  hl::WarpPolar warpFunc(param, centerX, centerY, maxRadius);
  warpFunc.schedule_cpu();

  warpFunc(input, output);

  auto res = hl::convertHalideToMat(output);
  cv::resize(res, res, {res.cols / 2, res.rows / 2});
  cv::rotate(res, res, cv::ROTATE_90_CLOCKWISE);

  cv::imwrite("warpPolar_halide.jpg", res);

  cv::imshow("", res);
  cv::waitKey(0);

  // Bench halide
  uspam::bench("warp 2D Halide", 100, [&] { warpFunc(input, output); });

  // Bench OpenCV
  cv::Mat cvMatOut;
  uspam::bench("warp 2D CV", 100, [&] {
    cv::warpPolar(img, cvMatOut, {outX, outY}, {centerX, centerY}, maxRadius,
                  cv::WARP_INVERSE_MAP | cv::WARP_FILL_OUTLIERS);
  });
}

void runWarp3D() {
  auto img = cv::imread("/Users/tnie/Downloads/stripes.jpg");
  const auto input = hl::convertMatToHalide(img);

  // Define the center, maximum radius, and whether it's a full circle
  const int r = std::min(input.width(), input.height());
  const int outX = r;
  const int outY = r;

  float centerX = static_cast<float>(r) / 2.0F;
  float centerY = static_cast<float>(r) / 2.0F;
  float maxRadius = std::min(centerX, centerY);

  Halide::ImageParam param(Halide::UInt(8), 3);

  Halide::Buffer<uint8_t> output(outX, outY, 3);
  hl::WarpPolar<3> warpFunc(param, centerX, centerY, maxRadius);
  warpFunc.schedule_cpu();

  warpFunc(input, output);

  auto res = hl::convertHalideToMat(output);
  cv::resize(res, res, {res.cols / 2, res.rows / 2});
  cv::imwrite("warpPolar_halide.jpg", res);

  cv::imshow("", res);
  cv::waitKey(0);
}

void runResize() {
  auto img =
      cv::imread("/Users/tnie/Downloads/stripes.jpg", cv::IMREAD_GRAYSCALE);
  const auto input = hl::convertMatToHalide(img);

  cv::Size dsize{200, 200};

  Halide::ImageParam param(Halide::UInt(8), 2);
  Halide::Buffer<uint8_t> output(dsize.width, dsize.height);

  hl::Resize resizeFunc(param, dsize.width, dsize.height);
  resizeFunc.schedule_cpu();
  resizeFunc(input, output);

  cv::Mat resizedImg = hl::convertHalideToMat(output);
  cv::imshow("", resizedImg);
  cv::waitKey(0);

  // Check correct
  {
    cv::Mat cvMatOut;
    cv::resize(img, cvMatOut, dsize);

    if (resizedImg.cols != dsize.width || resizedImg.rows != dsize.height) {
      fmt::println("resizedImg size incorrect");
      fmt::println("resizeImage size: ({}, {})", resizedImg.cols,
                   resizedImg.rows);
    }

    if (cvMatOut.cols != dsize.width || cvMatOut.rows != dsize.height) {
      fmt::println("cvMatOut size incorrect");
      fmt::println("cvMatOut size: ({}, {})", cvMatOut.cols, cvMatOut.rows);
    }

    int totalDiff = 0;
    for (int row = 0; row < resizedImg.rows; row++) {
      for (int col = 0; col < resizedImg.cols; col++) {
        const auto res = cvMatOut.at<uint8_t>(row, col);
        const auto expect = resizedImg.at<uint8_t>(row, col);
        if (res != expect) {
          fmt::println("Resize result incorrect at ({}, {}): got {}, expect {}",
                       col, row, res, expect);
          auto d = (int)cvMatOut.at<uint8_t>(col, row) -
                   (int)resizedImg.at<uint8_t>(col, row);
          totalDiff += std::abs(d);
        }
      }
    }
    fmt::println("Total diff: {}", totalDiff);
  }

  // Bench halide
  uspam::bench("Resize 2D Halide", 100, [&] { resizeFunc(input, output); });

  // Bench OpenCV
  {
    cv::Mat cvMatOut;
    uspam::bench("Resize 2D CV", 100,
                 [&] { cv::resize(img, cvMatOut, dsize); });
  }
}

int main(int argc, char *argv[]) {
  fmt::print("Hello, world!\n");

  // runHalide();

  fmt::println("Host target: {}", Halide::get_host_target().to_string());

  // runWarp2D();
  // runWarp3D();

  runResize();

  return 0;
}
