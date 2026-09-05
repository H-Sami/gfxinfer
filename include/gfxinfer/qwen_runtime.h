// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "gfxinfer/engine.h"
#include "gfxinfer/status.h"

namespace gfxinfer {

enum class ActivationMode {
  fp16,
  int8,
  int4,
  int4_w2,
  int4_w4,
};

struct DecodeSessionConfig {
  std::size_t maximum_context_tokens{4096};
  ActivationMode activation_mode{ActivationMode::fp16};
  // When nonzero, pre-instantiate the complete MTP proposal and verifier graph
  // for this target batch. Speculative depth D uses batch D + 1.
  unsigned graph_verifier_batch{0};
};

struct DecodeSessionSummary {
  std::size_t arena_bytes{0};
  std::size_t maximum_context_tokens{0};
  std::size_t gdn_state_bytes{0};
  std::size_t kv_cache_bytes{0};
  std::size_t mtp_cache_bytes{0};
};

struct SpeculativeStepResult {
  // Tokens made visible after the caller's seed token. The final entry is the
  // target model's correction (or bonus token after full acceptance).
  std::vector<std::uint32_t> tokens;
  std::size_t proposed_drafts{0};
  std::size_t accepted_drafts{0};
};

class QwenDecodeSession {
 public:
  ~QwenDecodeSession();
  QwenDecodeSession(const QwenDecodeSession&) = delete;
  QwenDecodeSession& operator=(const QwenDecodeSession&) = delete;
  QwenDecodeSession(QwenDecodeSession&&) noexcept;
  QwenDecodeSession& operator=(QwenDecodeSession&&) noexcept;

  [[nodiscard]] static Result<QwenDecodeSession> create(
      Engine& engine,
      DecodeSessionConfig config = {});
  [[nodiscard]] Result<std::uint32_t> greedy_step(std::uint32_t input_token);
  // Evaluate a known causal token block while loading each projection once.
  // This is the verifier primitive used by speculative decoding; state is
  // advanced by the entire block on success.
  [[nodiscard]] Result<std::vector<std::uint32_t>> block_step(
      std::span<const std::uint32_t> input_tokens);
  [[nodiscard]] Result<std::uint32_t> mtp_draft(std::uint32_t next_token);
  // Draft up to thirty-one future tokens with the built-in MTP layer, verify them
  // in one shared-weight target block, and commit only the accepted prefix.
  [[nodiscard]] Result<SpeculativeStepResult> speculative_step(
      std::uint32_t seed_token,
      std::size_t draft_count);
  [[nodiscard]] std::size_t position() const noexcept;
  [[nodiscard]] const DecodeSessionSummary& summary() const noexcept;

 private:
  struct Impl;
  explicit QwenDecodeSession(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gfxinfer
