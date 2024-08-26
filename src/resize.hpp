#pragma once

#include "halideUtils.hpp"
#include <Halide.h>
#include <fmt/core.h>
#include <utility>

namespace hl {
class Resize {
public:
  Resize(Halide::ImageParam input, float dwidth, float dheight)
      : input(std::move(input)), dwidth(dwidth), dheight(dheight) {
    setup();
  }

  void schedule_cpu() {
    Halide::Var xi("xi");
    Halide::Var yi("yi");

    resized.tile(x, y, xi, yi, 32, 32).parallel(y).vectorize(xi, 16);
    // resized.vectorize(x, 32).parallel(y);

    pipeline = Halide::Pipeline(resized);
    try {
      pipeline.compile_jit(Halide::get_host_target());
    } catch (const Halide::CompileError &e) {
      fmt::println("Halide exception: {}", e.what());
    }
  }

  void auto_schedule() {
    auto target = Halide::get_host_target();

    pipeline = Halide::Pipeline(resized);

    try {
      pipeline.compile_jit(Halide::get_host_target());
    } catch (const Halide::CompileError &e) {
      fmt::println("Halide exception: {}", e.what());
    }
  }

  template <typename T>
  void operator()(const Halide::Buffer<T> &input, Halide::Buffer<T> &output) {
    this->input.set(input);
    this->pipeline.realize(output);
  }

private:
  Halide::ImageParam input;
  float dwidth, dheight;

  Halide::Var x{"x"};
  Halide::Var y{"y"};

  Halide::Func resized;

  Halide::Pipeline pipeline;

  void setup() {
    using Halide::cast;
    using Halide::clamp;
    using Halide::floor;
    using Halide::lerp;

    auto scaleX = cast<float>(input.width()) / dwidth;
    auto scaleY = cast<float>(input.height()) / dheight;

    auto inX = (x + 0.5F) * scaleX - 0.5F;
    auto inY = (y + 0.5F) * scaleY - 0.5F;

    auto x0 = clamp(cast<int>(floor(inX)), 0, input.width() - 1);
    auto x1 = clamp(x0 + 1, 0, input.width() - 1);
    auto y0 = clamp(cast<int>(floor(inY)), 0, input.height() - 1);
    auto y1 = clamp(y0 + 1, 0, input.height() - 1);

    auto xf = frac(inX);
    auto yf = frac(inY);

    // Access the four neighboring pixels
    auto clamped = Halide::BoundaryConditions::repeat_edge(input);
    auto topLeft = clamped(x0, y0);
    auto topRight = clamped(x1, y0);
    auto bottomLeft = clamped(x0, y1);
    auto bottomRight = clamped(x1, y1);

    // Bilinear interpolation
    auto top = lerp(topLeft, topRight, xf);
    auto bottom = lerp(bottomLeft, bottomRight, xf);
    auto value = lerp(top, bottom, yf);

    // Save to output
    resized(x, y) = value;
  }
};

} // namespace hl