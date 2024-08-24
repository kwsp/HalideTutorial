#include <Halide.h>
#include <fmt/core.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "halideUtils.hpp"
#include "timeit.hpp"
#include "warpPolar.hpp"

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

void runWarp() {
  // Load an image
  auto img =
      cv::imread("/Users/tnie/Downloads/stripes.jpg", cv::IMREAD_GRAYSCALE);

  // cv::imshow("", img);
  // cv::waitKey(0);

  auto input = convertMatToHalide(img);
  fmt::println("Input channels: {}", input.channels());

  // Define the center, maximum radius, and whether it's a full circle
  const int r = std::min(input.width(), input.height());
  const int outX = r;
  const int outY = r;

  float centerX = static_cast<float>(r) / 2.0F;
  float centerY = static_cast<float>(r) / 2.0F;
  float maxRadius = std::min(centerX, centerY);

  // Define the polar warp function
  Halide::ImageParam param(Halide::UInt(8), 2);

  Halide::Buffer<uint8_t> output(outX, outY);
  warp::WarpPolar warpFunc(param, centerX, centerY, maxRadius);
  warpFunc.schedule_cpu();
  warpFunc(input, output);

  // Halide::Func polar_warped =
  //     warp::warp_polar(param, centerX, centerY, maxRadius);

  // // Scheduling: GPU target
  // Halide::Var x("x");
  // Halide::Var y("y");

  // // // Schedule
  // Halide::Var xi("xi");
  // Halide::Var yi("yi");
  // polar_warped.tile(x, y, xi, yi, 16, 16).vectorize(xi).parallel(y);

  // Halide::Pipeline pipeline(polar_warped);

  // // Set the target to use the GPU
  // Halide::Target target = Halide::get_host_target();
  // // target.set_feature(
  // //     Halide::Target::CUDA); // or Target::OpenCL, Target::Metal, etc.

  // pipeline.compile_jit(target);

  // // Realize the output on the GPU
  // Halide::Buffer<uint8_t> output(outX, outY);
  // param.set(input);
  // pipeline.realize(output);

  // {
  //   Halide::Buffer<uint8_t> output(outX, outY);
  //   param.set(input);
  //   pipeline.realize(output);
  // }

  auto res = convertHalideToMat(output);
  cv::resize(res, res, {res.cols / 2, res.rows / 2});

  cv::imwrite("warpPolar_halide.jpg", res);

  cv::imshow("", res);
  cv::waitKey(0);

  cv::Mat cvWarp;

  cv::warpPolar(img, cvWarp, {r, r}, {(float)r / 2, (float)r / 2},
                (double)r / 2, cv::WARP_INVERSE_MAP | cv::WARP_FILL_OUTLIERS);
  cv::resize(cvWarp, cvWarp, {cvWarp.cols / 2, cvWarp.rows / 2});
  cv::imwrite("warpPolar_cv.jpg", cvWarp);

  // cv::imshow("", cvWarp);
  // cv::waitKey(0);

  // Bench
  uspam::bench("warpPolar_halide", 100, [&] {
    // Halide::Buffer<uint8_t> output(outX, outY);
    // param.set(input);
    // pipeline.realize(output);

    warpFunc(input, output);
  });

  uspam::bench("warpPolar_cv", 100, [&] {
    cv::warpPolar(img, cvWarp, {r, r}, {(float)r / 2, (float)r / 2},
                  (double)r / 2, cv::WARP_INVERSE_MAP | cv::WARP_FILL_OUTLIERS);
  });
}

int main(int argc, char *argv[]) {
  fmt::print("Hello, world!\n");

  // runHalide();

  fmt::println("Host target: {}", Halide::get_host_target().to_string());

  runWarp();

  return 0;
}
