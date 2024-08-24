#pragma once

#include "halideUtils.hpp"
#include <Halide.h>
#include <utility>

namespace hl {
class Resize {
public:
  Resize(Halide::ImageParam input, float dwidth, float dheight)
      : input(std::move(input)), dwidth(dwidth), dheight(dheight) {
    setup();
  }

private:
  Halide::ImageParam input;
  float dwidth, dheight;

  Halide::Var x{"x"};
  Halide::Var y{"y"};

  Halide::Func resized;

  void setup() {
    auto scaleX = Halide::cast<float>(input.width()) / dwidth;
    auto scaleY = Halide::cast<float>(input.height()) / dheight;

    auto inX = (x + 0.5F) * scaleX - 0.5F;
    auto inY = (y + 0.5F) * scaleY - 0.5F;

    auto x0 = Halide::clamp(Halide::cast<int>(Halide::floor(inX)), 0,
                            input.width() - 1);
    auto x1 = Halide::clamp(x0 + 1, 0, input.width() - 1);
    auto y0 = Halide::clamp(Halide::cast<int>(Halide::floor(inY)), 0,
                            input.height() - 1);
    auto y1 = Halide::clamp(y0 + 1, 0, input.height() - 1);

    auto xf = frac(inX);
    auto yf = frac(inY);

    // Access the four neighboring pixels
    auto clamped = Halide::BoundaryConditions::repeat_edge(input);
    auto topLeft = clamped(x0, y0);
    auto topRight = clamped(x1, y0);
    auto bottomLeft = clamped(x0, y1);
    auto bottomRight = clamped(x1, y1);

    // Bilinear interpolation
    auto top = Halide::lerp(topLeft, topRight, xf);
    auto bottom = Halide::lerp(bottomLeft, bottomRight, xf);
    auto value = Halide::lerp(top, bottom, yf);

    // Save to output
    resized(x, y) = value;
  }
};

} // namespace hl