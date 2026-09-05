// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/tokenizer.h"

#include <simdjson.h>
#include <unicode/normalizer2.h>
#include <unicode/unistr.h>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gfxinfer/qwen38_spec.h"

namespace gfxinfer {
namespace {

[[nodiscard]] Status json_error(std::string_view context, simdjson::error_code error) {
  return {ErrorCode::corrupt_artifact,
          std::string(context) + ": " + simdjson::error_message(error)};
}

[[nodiscard]] bool decode_codepoint(
    std::string_view text,
    std::size_t& offset,
    std::uint32_t& codepoint) {
  if (offset >= text.size()) return false;
  const auto first = static_cast<std::uint8_t>(text[offset++]);
  if (first < 0x80U) {
    codepoint = first;
    return true;
  }
  unsigned continuation_count = 0;
  std::uint32_t value = 0;
  if ((first & 0xe0U) == 0xc0U) {
    continuation_count = 1;
    value = first & 0x1fU;
  } else if ((first & 0xf0U) == 0xe0U) {
    continuation_count = 2;
    value = first & 0x0fU;
  } else if ((first & 0xf8U) == 0xf0U) {
    continuation_count = 3;
    value = first & 0x07U;
  } else {
    return false;
  }
  if (offset + continuation_count > text.size()) return false;
  for (unsigned index = 0; index < continuation_count; ++index) {
    const auto byte = static_cast<std::uint8_t>(text[offset++]);
    if ((byte & 0xc0U) != 0x80U) return false;
    value = (value << 6U) | (byte & 0x3fU);
  }
  codepoint = value;
  return true;
}

[[nodiscard]] std::unordered_map<std::uint32_t, std::uint8_t> byte_decoder() {
  std::array<bool, 256> direct{};
  for (unsigned value = 33; value <= 126; ++value) direct[value] = true;
  for (unsigned value = 161; value <= 172; ++value) direct[value] = true;
  for (unsigned value = 174; value <= 255; ++value) direct[value] = true;
  std::unordered_map<std::uint32_t, std::uint8_t> decoder;
  decoder.reserve(256);
  std::uint32_t substitute = 256;
  for (std::uint32_t byte = 0; byte < 256; ++byte) {
    const auto codepoint = direct[byte] ? byte : substitute++;
    decoder.emplace(codepoint, static_cast<std::uint8_t>(byte));
  }
  return decoder;
}

void append_utf8(std::string& output, std::uint32_t codepoint) {
  if (codepoint < 0x80U) {
    output.push_back(static_cast<char>(codepoint));
  } else if (codepoint < 0x800U) {
    output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  } else if (codepoint < 0x10000U) {
    output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  } else {
    output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  }
}

[[nodiscard]] std::array<std::uint32_t, 256> byte_encoder() {
  std::array<bool, 256> direct{};
  for (unsigned value = 33; value <= 126; ++value) direct[value] = true;
  for (unsigned value = 161; value <= 172; ++value) direct[value] = true;
  for (unsigned value = 174; value <= 255; ++value) direct[value] = true;
  std::array<std::uint32_t, 256> encoder{};
  std::uint32_t substitute = 256;
  for (std::uint32_t byte = 0; byte < 256; ++byte)
    encoder[byte] = direct[byte] ? byte : substitute++;
  return encoder;
}

[[nodiscard]] std::string pair_key(std::string_view left, std::string_view right) {
  std::string key;
  key.reserve(left.size() + right.size() + 1);
  key.append(left);
  key.push_back('\0');
  key.append(right);
  return key;
}

[[nodiscard]] Result<std::string> normalize_nfc(std::string_view text) {
  UErrorCode error = U_ZERO_ERROR;
  const auto* normalizer = icu::Normalizer2::getNFCInstance(error);
  if (U_FAILURE(error) || normalizer == nullptr)
    return Status{ErrorCode::internal, "ICU NFC normalizer is unavailable"};
  icu::UnicodeString source = icu::UnicodeString::fromUTF8(text);
  icu::UnicodeString normalized;
  normalizer->normalize(source, normalized, error);
  if (U_FAILURE(error))
    return Status{ErrorCode::invalid_argument, "prompt is not valid normalizable UTF-8"};
  std::string output;
  normalized.toUTF8String(output);
  return output;
}

[[nodiscard]] Result<std::vector<std::string_view>> split_pretokens(
    const std::string& text) {
  static constexpr char pattern[] =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
  int error_code = 0;
  PCRE2_SIZE error_offset = 0;
  using CodePointer = std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)>;
  CodePointer code(
      pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern), PCRE2_ZERO_TERMINATED,
                    PCRE2_UTF | PCRE2_UCP, &error_code, &error_offset, nullptr),
      pcre2_code_free);
  if (!code) return Status{ErrorCode::internal, "cannot compile tokenizer pre-token regex"};
  using MatchPointer = std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)>;
  MatchPointer match(pcre2_match_data_create_from_pattern(code.get(), nullptr),
                     pcre2_match_data_free);
  if (!match) return Status{ErrorCode::out_of_memory, "cannot allocate tokenizer regex state"};
  std::vector<std::string_view> output;
  PCRE2_SIZE offset = 0;
  while (offset < text.size()) {
    const int result = pcre2_match(
        code.get(), reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), offset,
        PCRE2_ANCHORED, match.get(), nullptr);
    if (result < 1) {
      return Status{ErrorCode::invalid_argument,
                    "tokenizer pre-tokenizer could not consume the prompt"};
    }
    auto* vector = pcre2_get_ovector_pointer(match.get());
    if (vector[0] != offset || vector[1] <= offset) {
      return Status{ErrorCode::internal, "tokenizer regex returned an invalid range"};
    }
    output.emplace_back(text.data() + vector[0], vector[1] - vector[0]);
    offset = vector[1];
  }
  return output;
}

}  // namespace

Result<QwenTokenizer> QwenTokenizer::from_json(
    std::span<const std::byte> tokenizer_json) {
  if (tokenizer_json.empty()) {
    return Status{ErrorCode::invalid_argument, "tokenizer JSON resource is empty"};
  }
  simdjson::padded_string padded(
      reinterpret_cast<const char*>(tokenizer_json.data()), tokenizer_json.size());
  simdjson::dom::parser parser;
  simdjson::dom::element root;
  if (const auto error = parser.parse(padded).get(root); error)
    return json_error("cannot parse tokenizer JSON", error);

  QwenTokenizer tokenizer;
  tokenizer.pieces_.resize(Qwen38Spec::tokenizer_vocabulary_size);
  std::vector<bool> observed(Qwen38Spec::tokenizer_vocabulary_size, false);
  simdjson::dom::object vocabulary;
  if (const auto error = root["model"]["vocab"].get(vocabulary); error)
    return json_error("tokenizer model.vocab is invalid", error);
  for (const auto field : vocabulary) {
    std::int64_t id = -1;
    if (const auto error = field.value.get(id); error)
      return json_error("tokenizer vocabulary ID is invalid", error);
    if (id < 0 || id >= static_cast<std::int64_t>(tokenizer.pieces_.size()) ||
        observed[static_cast<std::size_t>(id)]) {
      return Status{ErrorCode::corrupt_artifact, "tokenizer vocabulary ID is out of range or duplicated"};
    }
    tokenizer.pieces_[static_cast<std::size_t>(id)] = std::string(field.key);
    tokenizer.token_ids_.emplace(std::string(field.key), static_cast<std::uint32_t>(id));
    observed[static_cast<std::size_t>(id)] = true;
  }
  simdjson::dom::array added_tokens;
  if (const auto error = root["added_tokens"].get(added_tokens); error)
    return json_error("tokenizer added_tokens is invalid", error);
  for (const auto element : added_tokens) {
    std::int64_t id = -1;
    std::string_view content;
    if (const auto error = element["id"].get(id); error)
      return json_error("added-token ID is invalid", error);
    if (const auto error = element["content"].get(content); error)
      return json_error("added-token content is invalid", error);
    if (id < 0 || id >= static_cast<std::int64_t>(tokenizer.pieces_.size())) {
      return Status{ErrorCode::corrupt_artifact, "added-token ID is out of range"};
    }
    tokenizer.pieces_[static_cast<std::size_t>(id)] = std::string(content);
    tokenizer.token_ids_[std::string(content)] = static_cast<std::uint32_t>(id);
    tokenizer.added_tokens_.emplace_back(std::string(content), static_cast<std::uint32_t>(id));
    observed[static_cast<std::size_t>(id)] = true;
  }
  for (std::size_t id = 0; id < observed.size(); ++id) {
    if (!observed[id]) {
      return Status{ErrorCode::corrupt_artifact,
                    "tokenizer has no piece for ID " + std::to_string(id)};
    }
  }
  simdjson::dom::array merges;
  if (const auto error = root["model"]["merges"].get(merges); error)
    return json_error("tokenizer model.merges is invalid", error);
  std::uint32_t rank = 0;
  for (const auto element : merges) {
    std::string_view merge;
    if (const auto error = element.get(merge); error)
      return json_error("tokenizer merge is invalid", error);
    const auto separator = merge.find(' ');
    if (separator == std::string_view::npos || separator == 0 || separator + 1 >= merge.size()) {
      return Status{ErrorCode::corrupt_artifact, "tokenizer merge pair is malformed"};
    }
    tokenizer.merge_ranks_.emplace(
        pair_key(merge.substr(0, separator), merge.substr(separator + 1)), rank++);
  }
  std::sort(tokenizer.added_tokens_.begin(), tokenizer.added_tokens_.end(),
            [](const auto& left, const auto& right) {
              return left.first.size() > right.first.size();
            });
  return tokenizer;
}

Result<std::string> QwenTokenizer::decode(
    std::span<const std::uint32_t> tokens) const {
  const auto decoder = byte_decoder();
  std::string output;
  for (const auto token : tokens) {
    if (token >= pieces_.size()) {
      return Status{ErrorCode::invalid_argument,
                    "token ID exceeds the tokenizer vocabulary"};
    }
    const auto& piece = pieces_[token];
    std::size_t offset = 0;
    while (offset < piece.size()) {
      std::uint32_t codepoint = 0;
      if (!decode_codepoint(piece, offset, codepoint)) {
        return Status{ErrorCode::corrupt_artifact, "tokenizer piece contains invalid UTF-8"};
      }
      const auto found = decoder.find(codepoint);
      if (found == decoder.end()) {
        return Status{ErrorCode::corrupt_artifact,
                      "tokenizer piece is outside the ByteLevel alphabet"};
      }
      output.push_back(static_cast<char>(found->second));
    }
  }
  return output;
}

Result<std::vector<std::uint32_t>> QwenTokenizer::encode(std::string_view text) const {
  const auto encoder = byte_encoder();
  std::vector<std::uint32_t> output;
  auto encode_normal = [&](std::string_view segment) -> Status {
    if (segment.empty()) return Status::ok_status();
    auto normalized = normalize_nfc(segment);
    if (!normalized.ok()) return normalized.status();
    auto pretokens = split_pretokens(normalized.value());
    if (!pretokens.ok()) return pretokens.status();
    for (const auto pretoken : pretokens.value()) {
      std::string encoded;
      encoded.reserve(pretoken.size() * 2);
      for (const unsigned char byte : pretoken) append_utf8(encoded, encoder[byte]);
      std::vector<std::string> symbols;
      for (std::size_t offset = 0; offset < encoded.size();) {
        const auto begin = offset;
        std::uint32_t codepoint = 0;
        if (!decode_codepoint(encoded, offset, codepoint))
          return {ErrorCode::internal, "ByteLevel encoder produced invalid UTF-8"};
        symbols.emplace_back(encoded.substr(begin, offset - begin));
      }
      while (symbols.size() > 1) {
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::string best_pair;
        for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
          const auto key = pair_key(symbols[index], symbols[index + 1]);
          const auto found = merge_ranks_.find(key);
          if (found != merge_ranks_.end() && found->second < best_rank) {
            best_rank = found->second;
            best_pair = key;
          }
        }
        if (best_pair.empty()) break;
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (std::size_t index = 0; index < symbols.size();) {
          if (index + 1 < symbols.size() &&
              pair_key(symbols[index], symbols[index + 1]) == best_pair) {
            merged.push_back(symbols[index] + symbols[index + 1]);
            index += 2;
          } else {
            merged.push_back(std::move(symbols[index++]));
          }
        }
        symbols = std::move(merged);
      }
      for (const auto& symbol : symbols) {
        const auto found = token_ids_.find(symbol);
        if (found == token_ids_.end())
          return {ErrorCode::corrupt_artifact, "BPE output is absent from the vocabulary"};
        output.push_back(found->second);
      }
    }
    return Status::ok_status();
  };

  std::size_t cursor = 0;
  while (cursor < text.size()) {
    std::size_t special_position = std::string_view::npos;
    const std::pair<std::string, std::uint32_t>* special = nullptr;
    for (const auto& candidate : added_tokens_) {
      const auto found = text.find(candidate.first, cursor);
      if (found < special_position) {
        special_position = found;
        special = &candidate;
      }
    }
    if (special == nullptr) {
      auto status = encode_normal(text.substr(cursor));
      if (!status.ok()) return status;
      break;
    }
    auto status = encode_normal(text.substr(cursor, special_position - cursor));
    if (!status.ok()) return status;
    output.push_back(special->second);
    cursor = special_position + special->first.size();
  }
  return output;
}

}  // namespace gfxinfer
