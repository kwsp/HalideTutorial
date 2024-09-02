#pragma once

#include "halideUtils.hpp"
#include <Halide.h>
#include <fmt/core.h>
#include <utility>

namespace hl {

template <int Dims = 2> class Resize {
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

  bool schedule_gpu() {
    auto target = find_gpu_target();
    if (!target.has_gpu_feature()) {
      return false;
    }

    Halide::Var block;
    Halide::Var thread;

    Halide::Var xi("xi");
    Halide::Var yi("yi");

    resized.gpu_tile(x, y, xi, yi, 32, 32).parallel(y).vectorize(xi, 16);

    resized.gpu_threads(y);

    pipeline = Halide::Pipeline(resized);
    try {
      pipeline.compile_jit(Halide::get_host_target());
    } catch (const Halide::CompileError &e) {
      fmt::println("Halide exception: {}", e.what());
    }
    return true;
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
  Halide::Var c{"c"};

  Halide::Func resized;

  Halide::Pipeline pipeline;

  void setup() {
    using Halide::cast;

    auto scaleX = cast<float>(input.width()) / dwidth;
    auto scaleY = cast<float>(input.height()) / dheight;

    auto inX = (x + 0.5F) * scaleX - 0.5F;
    auto inY = (y + 0.5F) * scaleY - 0.5F;

    if constexpr (Dims == 2) {
      resized(x, y) = bilinear_interpolate<Dims>(input, inX, inY);
    } else {
      resized(x, y) = bilinear_interpolate<Dims>(input, inX, inY, c);
    }
  }
};

} // namespace hl