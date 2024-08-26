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

int runHalide() {
  // This program defines a single-stage imaging pipeline that
  // outputs a grayscale diagonal gradient.

  // A 'Func' object represents a pipeline stage. It's a pure
  // function that defines what value each pixel should have. You
  // can think of it as a computed image.
  Halide::Func gradient;

  // Var objects are names to use as variables in the definition of
  // a Func. They have no meaning by themselves.
  Halide::Var x, y;

  // We typically use Vars named 'x' and 'y' to correspond to the x
  // and y axes of an image, and we write them in that order. If
  // you're used to thinking of images as having rows and columns,
  // then x is the column index, and y is the row index.

  // Funcs are defined at any integer coordinate of its variables as
  // an Expr in terms of those variables and other functions.
  // Here, we'll define an Expr which has the value x + y. Vars have
  // appropriate operator overloading so that expressions like
  // 'x + y' become 'Expr' objects.
  Halide::Expr e = x + y;

  // Now we'll add a definition for the Func object. At pixel x, y,
  // the image will have the value of the Expr e. On the left hand
  // side we have the Func we're defining and some Vars. On the right
  // hand side we have some Expr object that uses those same Vars.
  gradient(x, y) = e;

  // This is the same as writing:
  //
  //   gradient(x, y) = x + y;
  //
  // which is the more common form, but we are showing the
  // intermediate Expr here for completeness.

  // That line of code defined the Func, but it didn't actually
  // compute the output image yet. At this stage it's just Funcs,
  // Exprs, and Vars in memory, representing the structure of our
  // imaging pipeline. We're meta-programming. This C++ program is
  // constructing a Halide program in memory. Actually computing
  // pixel data comes next.

  // Now we 'realize' the Func, which JIT compiles some code that
  // implements the pipeline we've defined, and then runs it.  We
  // also need to tell Halide the domain over which to evaluate the
  // Func, which determines the range of x and y above, and the
  // resolution of the output image. Halide.h also provides a basic
  // templatized image type we can use. We'll make an 800 x 600
  // image.
  Halide::Buffer<int32_t> output = gradient.realize({800, 600});

  // Halide does type inference for you. Var objects represent
  // 32-bit integers, so the Expr object 'x + y' also represents a
  // 32-bit integer, and so 'gradient' defines a 32-bit image, and
  // so we got a 32-bit signed integer image out when we call
  // 'realize'. Halide types and type-casting rules are equivalent
  // to C.

  // Let's check everything worked, and we got the output we were
  // expecting:
  for (int j = 0; j < output.height(); j++) {
    for (int i = 0; i < output.width(); i++) {
      // We can access a pixel of an Buffer object using similar
      // syntax to defining and using functions.
      if (output(i, j) != i + j) {
        printf("Something went wrong!\n"
               "Pixel %d, %d was supposed to be %d, but instead it's %d\n",
               i, j, i + j, output(i, j));
        return -1;
      }
    }
  }

  // Everything worked! We defined a Func, then called 'realize' on
  // it to generate and run machine code that produced an Buffer.
  printf("Success!\n");
  return 0;
}

// void medianBlur(Halide::Buffer<uint8_t> input ){
// Halide::Var x("x");
// Halide::Var y("y");

// Halide::Func f;
// Halide::RDom r(-2, 5, -2, 5);

// }

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
