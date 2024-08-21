set(EXE_NAME main)

find_package(fmt CONFIG REQUIRED)

find_package(Armadillo CONFIG REQUIRED)
find_package(Halide CONFIG REQUIRED)

add_executable(${EXE_NAME}
  main.cpp
)

set_target_properties(${EXE_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

target_link_libraries(${EXE_NAME} PRIVATE
  fmt::fmt
  armadillo
  Halide::Halide
)