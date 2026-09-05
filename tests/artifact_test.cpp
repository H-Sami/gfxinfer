// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/artifact.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

template <typename T>
std::span<const std::byte> bytes(const std::vector<T>& values) {
  return {reinterpret_cast<const std::byte*>(values.data()), values.size() * sizeof(T)};
}

bool expect(bool condition, const char* message) {
  if (!condition) std::cerr << "artifact test: " << message << '\n';
  return condition;
}

}  // namespace

bool artifact_tests() {
  const auto test_directory = std::filesystem::path(GFXINFER_TEST_OUTPUT_DIR);
  std::error_code ignored;
  std::filesystem::create_directories(test_directory, ignored);
  const auto path = test_directory / "gfxinfer-artifact-test.gfxi";
  const std::vector<std::uint8_t> codes{0x12, 0x34, 0x56, 0x78};
  const std::vector<float> scales{0.25F, 0.5F};
  const std::array requests{
      gfxinfer::TensorWriteRequest{
          "text/layers/0/mlp/gate_up", gfxinfer::DType::u8,
          gfxinfer::Quantization::w2g128, {2, 128}, bytes(codes), bytes(scales), 256},
      gfxinfer::TensorWriteRequest{
          "text/final_norm", gfxinfer::DType::f32,
          gfxinfer::Quantization::none, {2}, bytes(scales), {}, 256},
  };
  const gfxinfer::ArtifactIdentity identity{
      "gfx1200", "qwen3.8-27b", "test-weights", "test-recipe-v1"};
  auto status = gfxinfer::ArtifactWriter::write(path, identity, requests);
  if (!expect(status.ok(), status.message().c_str())) return false;

  auto artifact = gfxinfer::ArtifactView::open(path);
  bool passed = expect(artifact.ok(), artifact.ok() ? "" : artifact.status().message().c_str());
  if (artifact.ok()) {
    passed = expect(artifact.value().identity().target_arch == "gfx1200", "target identity") && passed;
    passed = expect(artifact.value().identity().model_id == "qwen3.8-27b", "model identity") && passed;
    passed = expect(artifact.value().tensors().size() == 2, "tensor count") && passed;
    const auto* tensor = artifact.value().find("text/layers/0/mlp/gate_up");
    passed = expect(tensor != nullptr, "tensor lookup") && passed;
    if (tensor != nullptr) {
      const auto payload = artifact.value().data(*tensor);
      passed = expect(payload.size() == codes.size(), "payload size") && passed;
      passed = expect(std::to_integer<std::uint8_t>(payload[2]) == codes[2], "payload bytes") && passed;
    }
  }
  std::filesystem::remove(path, ignored);

  const std::array duplicates{
      gfxinfer::TensorWriteRequest{"duplicate", gfxinfer::DType::u8,
                                   gfxinfer::Quantization::none, {4}, bytes(codes), {}, 256},
      gfxinfer::TensorWriteRequest{"duplicate", gfxinfer::DType::u8,
                                   gfxinfer::Quantization::none, {4}, bytes(codes), {}, 256},
  };
  status = gfxinfer::ArtifactWriter::write(path, identity, duplicates);
  passed = expect(!status.ok(), "duplicate tensor names must be rejected") && passed;

  const auto stream_path = test_directory / "gfxinfer-stream-test.gfxi";
  const std::array plans{
      gfxinfer::TensorWritePlan{"stream/codes", gfxinfer::DType::u8,
                                gfxinfer::Quantization::w2g128, {2, 128}, 4, 8, 256},
      gfxinfer::TensorWritePlan{"stream/norm", gfxinfer::DType::f32,
                                gfxinfer::Quantization::none, {2}, 8, 0, 256},
  };
  auto stream = gfxinfer::ArtifactStreamWriter::create(stream_path, identity, plans);
  passed = expect(stream.ok(), stream.ok() ? "" : stream.status().message().c_str()) && passed;
  if (stream.ok()) {
    const auto code_bytes = bytes(codes);
    const auto scale_bytes = bytes(scales);
    status = stream.value().append_data(0, code_bytes.first(2));
    passed = expect(status.ok(), "first streamed code chunk") && passed;
    status = stream.value().append_data(0, code_bytes.last(2));
    passed = expect(status.ok(), "second streamed code chunk") && passed;
    status = stream.value().append_auxiliary(0, scale_bytes);
    passed = expect(status.ok(), "streamed scale plane") && passed;
    status = stream.value().append_data(1, scale_bytes);
    passed = expect(status.ok(), "streamed norm") && passed;
    status = stream.value().finalize();
    passed = expect(status.ok(), status.message().c_str()) && passed;
  }
  auto streamed = gfxinfer::ArtifactView::open(stream_path);
  passed = expect(streamed.ok(), streamed.ok() ? "" : streamed.status().message().c_str()) && passed;
  if (streamed.ok()) {
    const auto* tensor = streamed.value().find("stream/codes");
    passed = expect(tensor != nullptr && streamed.value().data(*tensor).size() == 4,
                    "streamed artifact payload") && passed;
  }
  std::filesystem::remove(stream_path, ignored);
  return passed;
}
