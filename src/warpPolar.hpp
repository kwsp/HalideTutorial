#pragma once

#include <Halide.h>
#include <numbers>
#include <stdexcept>

namespace warp {

template <int Dims = 2> class WarpPolar {
public:
  WarpPolar(Halide::ImageParam input, float centerX, float centerY,
            float maxRadius)
      : m_input(std::move(input)) {
    // Compute the maximum angle, 360 degrees or 180 degrees depending on
    // fullCircle
    const float pi = std::numbers::pi;
    float maxAngle = 2.0f * 2 * pi;

    // Compute polar coordinates (radius, angle) for each point
    Halide::Expr dx = x - centerX;
    Halide::Expr dy = y - centerY;
    Halide::Expr radius = Halide::sqrt(dx * dx + dy * dy);
    Halide::Expr angle = Halide::atan2(dy, dx);

    // Normalize the radius and angle
    Halide::Expr normalized_radius = radius / maxRadius;
    Halide::Expr normalized_angle = (angle + pi) / maxAngle;

    // Calculate corresponding Cartesian coordinates in the input image
    Halide::Expr srcX =
        Halide::clamp(Halide::cast<int>(normalized_radius * m_input.width()), 0,
                      m_input.width() - 1);
    Halide::Expr srcY =
        Halide::clamp(Halide::cast<int>(normalized_angle * m_input.height()), 0,
                      m_input.height() - 1);

    // Create the output function
    if constexpr (Dims == 2) {
      // 2D grayscale image
      warpped(x, y) = m_input(srcX, srcY);
    } else if constexpr (Dims == 3) {
      Halide::Var c("c");
      warpped(x, y, c) = m_input(srcX, srcY, c);
    } else {
      static_assert(Dims == 2 || Dims == 3, "Unsupported number of dimensions");
    }
  }

  void schedule_cpu() {
    // Scheduling: GPU target
    Halide::Var xi("xi");
    Halide::Var yi("yi");
    warpped.tile(x, y, xi, yi, 16, 16).vectorize(xi, 8).parallel(y);

    pipeline = Halide::Pipeline(warpped);
    pipeline.compile_jit(Halide::get_host_target());
  }

  template <typename T>
  void operator()(const Halide::Buffer<T> &input, Halide::Buffer<T> &output) {
    m_input.set(input);
    pipeline.realize(output);
  }

private:
  Halide::ImageParam m_input;
  Halide::Var x{"x"};
  Halide::Var y{"y"};

  Halide::Func warpped{"warpped"};

  Halide::Pipeline pipeline;
};

} // namespace warp
