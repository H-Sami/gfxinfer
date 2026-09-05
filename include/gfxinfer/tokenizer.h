// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gfxinfer/status.h"

namespace gfxinfer {

class QwenTokenizer {
 public:
  [[nodiscard]] static Result<QwenTokenizer> from_json(
      std::span<const std::byte> tokenizer_json);
  [[nodiscard]] Result<std::string> decode(
      std::span<const std::uint32_t> tokens) const;
  [[nodiscard]] Result<std::vector<std::uint32_t>> encode(
      std::string_view text) const;
  [[nodiscard]] std::size_t vocabulary_size() const noexcept { return pieces_.size(); }

 private:
  std::vector<std::string> pieces_;
  std::unordered_map<std::string, std::uint32_t> token_ids_;
  std::unordered_map<std::string, std::uint32_t> merge_ranks_;
  std::vector<std::pair<std::string, std::uint32_t>> added_tokens_;
};

}  // namespace gfxinfer
