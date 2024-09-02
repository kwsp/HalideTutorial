#pragma once

#include "halideUtils.hpp"
#include <Halide.h>
#include <fmt/core.h>
#include <numbers>

namespace hl {
enum Direction { Forward, Backward };

template <int Dims = 2> class WarpPolar {
public:
  WarpPolar(Halide::ImageParam input, float centerX, float centerY,
            float maxRadius)
      : input(std::move(input)), centerX(centerX), centerY(centerY),
        maxRadius(maxRadius) {
    static_assert(Dims == 2 || Dims == 3, "Unsupported number of dimensions");
    setupFuncInverse();
  }

  void schedule_cpu() {
    Halide::Var xi("xi");
    Halide::Var yi("yi");
    Halide::Var ci("ci");

    if constexpr (Dims == 2) {
      warpped.tile(x, y, xi, yi, 16, 16).vectorize(xi, 8).parallel(y);
    } else if constexpr (Dims == 3) {
      warpped.reorder(c, x, y)
          .tile(x, y, xi, yi, 16, 16)
          .vectorize(xi, 8)
          .parallel(y)
          .unroll(c, 3);
      // warpped.compute_root();
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
    constexpr float PI = std::numbers::pi_v<float>;
    constexpr float maxAngle = 2.0F * PI;
    using Halide::atan2;
    using Halide::cast;
    using Halide::clamp;
    using Halide::Expr;
    using Halide::sqrt;

    // Compute polar coordinates (radius, angle) for each point
    Expr dx = cast<float>(x) - centerX;
    Expr dy = cast<float>(y) - centerY;
    Expr radius = sqrt(dx * dx + dy * dy);
    Expr angle = atan2(dy, dx);

    // Normalize the radius and angle
    auto normalized_radius = radius / maxRadius;
    auto normalized_angle = (angle + PI) / maxAngle;

    // Calculate corresponding Cartesian coordinates in the input image
    auto srcX = normalized_radius * input.width();
    auto srcY = normalized_angle * input.height();
    // Only check X bound here. When Y bound is checked, there is an error in
    // indexing where the 0th degree is somehow out of bounds.
    auto outOfBound = srcX < 0 || srcX >= input.width();

    // Expr srcXclamped = clamp(cast<int>(srcX), 0, input.width() - 1);
    // Expr srcYclamped = clamp(cast<int>(srcY), 0, input.height() - 1);
    // value = input(srcXclamped, srcYclamped);

    // Bilinear interpolation
    if constexpr (Dims == 2) {
      warpped(x, y) =
          select(outOfBound, 0, bilinear_interpolate<Dims>(input, srcX, srcY));

    } else {
      warpped(x, y, c) = select(
          outOfBound, 0, bilinear_interpolate<Dims>(input, srcX, srcY, c));
    }

    // warpped(x, y) = select(outOfBound, fillValue, cast(input.type(), 255));
  }
};

} // namespace hl
