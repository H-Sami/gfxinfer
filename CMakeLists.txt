cmake_minimum_required(VERSION 3.24)

project(gfxinfer VERSION 0.1.0 LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

if(NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1200)
endif()

option(GFXINFER_BUILD_TESTS "Build the CPU-side validation suite" ON)
option(GFXINFER_ENABLE_SANITIZERS "Enable ASan and UBSan for host code" OFF)

find_package(hip CONFIG REQUIRED)
find_package(simdjson CONFIG REQUIRED)
find_package(OpenMP)
find_package(ICU REQUIRED COMPONENTS uc i18n)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)

add_library(gfxinfer_core STATIC
  src/artifact.cpp
  src/converter.cpp
  src/device.cpp
  src/engine.cpp
  src/quantization.cpp
  src/qwen38_spec.cpp
  src/qwen_runtime.cpp
  src/safetensors.cpp
  src/tokenizer.cpp
)
target_include_directories(gfxinfer_core PUBLIC include)
target_link_libraries(gfxinfer_core PUBLIC
  hip::host
  simdjson::simdjson
  ICU::uc
  ICU::i18n
  PkgConfig::PCRE2
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gfxinfer_core PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_options(gfxinfer_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>
)

add_library(gfxinfer_kernels STATIC
  src/kernels/lowbit_gemv.hip
  src/kernels/qwen_ops.hip
)
target_include_directories(gfxinfer_kernels PUBLIC include)
target_link_libraries(gfxinfer_kernels PUBLIC hip::host)
target_compile_options(gfxinfer_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-Wall;-Wextra>
)

if(GFXINFER_ENABLE_SANITIZERS)
  target_compile_options(gfxinfer_core PRIVATE -fsanitize=address,undefined)
  target_link_options(gfxinfer_core PRIVATE -fsanitize=address,undefined)
endif()

add_executable(gfxinfer src/cli/main.cpp)
target_link_libraries(gfxinfer PRIVATE gfxinfer_core gfxinfer_kernels)

add_executable(gfxinfer-server src/server/main.cpp)
target_link_libraries(gfxinfer-server PRIVATE gfxinfer_core gfxinfer_kernels)
target_compile_options(gfxinfer-server PRIVATE -Wall -Wextra -Wpedantic)

include(GNUInstallDirs)
install(TARGETS gfxinfer gfxinfer-server RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(GFXINFER_BUILD_TESTS)
  enable_testing()
  add_executable(gfxinfer_tests
    tests/artifact_test.cpp
    tests/quantization_test.cpp
    tests/qwen38_spec_test.cpp
    tests/safetensors_test.cpp
    tests/test_main.cpp
  )
  target_link_libraries(gfxinfer_tests PRIVATE gfxinfer_core)
  target_compile_definitions(gfxinfer_tests PRIVATE
    GFXINFER_TEST_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/test-artifacts"
  )
  add_test(NAME gfxinfer_tests COMMAND gfxinfer_tests)
endif()
