#pragma once

#include <Halide.h>
#include <fmt/core.h>
#include <numbers>

namespace warp {

template <int Dims = 2> class WarpPolar {
public:
  WarpPolar(Halide::ImageParam input, float centerX, float centerY,
            float maxRadius)
      : input(std::move(input)), centerX(centerX), centerY(centerY),
        maxRadius(maxRadius) {
    setupFuncInverse();
  }

  void schedule_cpu() {
    // Scheduling: GPU target
    Halide::Var xi("xi");
    Halide::Var yi("yi");
    Halide::Var ci("ci");

    if constexpr (Dims == 2) {
      warpped.tile(x, y, xi, yi, 16, 16).vectorize(xi, 8).parallel(y);
    } else if constexpr (Dims == 3) {
      // warpped.reorder(c, x, y)
      //     .tile(x, y, xi, yi, 16, 16)
      //     .vectorize(xi, 8)
      //     .parallel(y)
      //     .unroll(c, 3);
      warpped.compute_root();
    }

    pipeline = Halide::Pipeline(warpped);
    try {
      pipeline.compile_jit(Halide::get_host_target());
    } catch (const Halide::CompileError &e) {
      fmt::println("Halide exception: {}", e.what());
      throw e;
    }
  }

  template <typename T>
  void operator()(const Halide::Buffer<T> &input, Halide::Buffer<T> &output) {
    this->input.set(input);
    this->pipeline.realize(output);
  }

private:
  Halide::ImageParam input;
  Halide::Var x{"x"};
  Halide::Var y{"y"};
  Halide::Var c{"c"};

  float centerX, centerY;
  float maxRadius;

  Halide::Func warpped{"warpped"};

  Halide::Pipeline pipeline;

  void setupFuncInverse() {
    // Compute the maximum angle, 360 degrees or 180 degrees depending on
    // fullCircle
    float maxAngle = 2.0F * std::numbers::pi_v<float>;

    // Compute polar coordinates (radius, angle) for each point
    Halide::Expr dx = x - centerX;
    Halide::Expr dy = y - centerY;
    Halide::Expr radius = Halide::sqrt(dx * dx + dy * dy);
    Halide::Expr angle = Halide::atan2(dy, dx);

    // Normalize the radius and angle
    Halide::Expr normalized_radius = radius / maxRadius;
    Halide::Expr normalized_angle =
        (angle + std::numbers::pi_v<float>) / maxAngle;

    // Calculate corresponding Cartesian coordinates in the input image
    Halide::Expr srcX =
        Halide::clamp(Halide::cast<int>(normalized_radius * input.width()), 0,
                      input.width() - 1);
    Halide::Expr srcY =
        Halide::clamp(Halide::cast<int>(normalized_angle * input.height()), 0,
                      input.height() - 1);

    Halide::Expr outOfBound =
        srcX < 0 || srcX >= input.width() || srcY < 0 || srcY >= input.height();

    Halide::Expr fillValue = Halide::cast(input.type(), 0);

    // Create the output function
    if constexpr (Dims == 2) {
      // 2D grayscale image
      warpped(x, y) = input(srcX, srcY);
      // warpped(x, y) = Halide::select(outOfBound, fillValue, input(srcX,
      // srcY));
    } else if constexpr (Dims == 3) {
      warpped(x, y, c) = input(srcX, srcY, c);
    } else {
      static_assert(Dims == 2 || Dims == 3, "Unsupported number of dimensions");
    }
  }

  Halide::Expr frac(const Halide::Expr &x) { return x - Halide::floor(x); }
};

} // namespace warp
