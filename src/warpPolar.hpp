#pragma once

#include <Halide.h>
#include <fmt/core.h>
#include <numbers>

namespace warp {

template <int Dims = 2> class WarpPolar {
public:
  WarpPolar(Halide::ImageParam input, float centerX, float centerY,
            float maxRadius)
      : m_input(std::move(input)), centerX(centerX), centerY(centerY),
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
    m_input.set(input);
    pipeline.realize(output);
  }

private:
  Halide::ImageParam m_input;
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
        Halide::clamp(Halide::cast<int>(normalized_radius * m_input.width()), 0,
                      m_input.width() - 1);
    Halide::Expr srcY =
        Halide::clamp(Halide::cast<int>(normalized_angle * m_input.height()), 0,
                      m_input.height() - 1);

    Halide::Expr outOfBound = srcX < 0 || srcX >= m_input.width() || srcY < 0 ||
                              srcY >= m_input.height();

    Halide::Expr fillValue = Halide::cast(m_input.type(), 0);

    // Create the output function
    if constexpr (Dims == 2) {
      // 2D grayscale image
      // warpped(x, y) = m_input(srcX, srcY);
      warpped(x, y) =
          Halide::select(outOfBound, fillValue, m_input(srcX, srcY));
    } else if constexpr (Dims == 3) {
      warpped(x, y, c) = m_input(srcX, srcY, c);
    } else {
      static_assert(Dims == 2 || Dims == 3, "Unsupported number of dimensions");
    }
  }

  Halide::Expr frac(const Halide::Expr &x) { return x - Halide::floor(x); }

  void setupFuncForward() {

    // Compute the maximum angle, 360 degrees or 180 degrees depending on
    // fullCircle
    float maxAngle = 2.0F * std::numbers::pi;

    // Compute polar coordinates based on the output pixel location
    Halide::Expr radius = Halide::cast<float>(y) * maxRadius / m_input.height();
    Halide::Expr angle = Halide::cast<float>(x) * maxAngle / m_input.width() -
                         std::numbers::pi_v<float>;

    // Convert polar coordinates to Cartesian coordinates
    Halide::Expr srcX = centerX + radius * cos(angle);
    Halide::Expr srcY = centerY + radius * sin(angle);

    // Check if coordinates are out of bounds
    Halide::Expr out_of_bounds = srcX < 0 || srcX >= m_input.width() ||
                                 srcY < 0 || srcY >= m_input.height();

    // Define the fill value for outliers using the same type as the input image
    Halide::Type input_type = m_input.type();
    Halide::Expr fill_value = cast(input_type, 0);

    // Use bilinear interpolation for better quality
    Halide::Func interpolated =
        Halide::BoundaryConditions::repeat_edge(m_input);
    Halide::Expr v1 =
        interpolated(clamp(cast<int>(srcX), 0, m_input.width() - 1),
                     clamp(cast<int>(srcY), 0, m_input.height() - 1));
    Halide::Expr v2 =
        interpolated(clamp(cast<int>(srcX) + 1, 0, m_input.width() - 1),
                     clamp(cast<int>(srcY), 0, m_input.height() - 1));
    Halide::Expr v3 =
        interpolated(clamp(cast<int>(srcX), 0, m_input.width() - 1),
                     clamp(cast<int>(srcY) + 1, 0, m_input.height() - 1));
    Halide::Expr v4 =
        interpolated(clamp(cast<int>(srcX) + 1, 0, m_input.width() - 1),
                     clamp(cast<int>(srcY) + 1, 0, m_input.height() - 1));

    Halide::Expr xf = frac(srcX);
    Halide::Expr yf = frac(srcY);
    Halide::Expr interpolated_value =
        lerp(lerp(v1, v2, xf), lerp(v3, v4, xf), yf);

    // Create the output function
    warpped(x, y) = interpolated_value;
    // warpped(x, y) =
    //     Halide::select(out_of_bounds, fill_value, interpolated_value);
  }
};

} // namespace warp
