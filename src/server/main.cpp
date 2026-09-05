// SPDX-License-Identifier: Apache-2.0

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <simdjson.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gfxinfer/engine.h"
#include "gfxinfer/qwen38_spec.h"
#include "gfxinfer/qwen_runtime.h"
#include "gfxinfer/status.h"
#include "gfxinfer/tokenizer.h"
#include "gfxinfer/version.h"

namespace {

constexpr std::string_view kModelId = "qwen3.8-27b-gfx1200";
constexpr std::uint32_t kEndToken = 248044U;
constexpr std::uint32_t kEndOfTextToken = 248046U;
constexpr std::size_t kMaximumHeaderBytes = 64U << 10U;
constexpr std::size_t kMaximumBodyBytes = 1U << 20U;

struct ServerOptions {
  std::string artifact;
  std::string host{"127.0.0.1"};
  std::uint16_t port{8000};
  std::size_t context{4096};
  std::size_t maximum_output_tokens{1024};
  std::size_t draft_count{4};
  gfxinfer::ActivationMode activation{gfxinfer::ActivationMode::int4};
};

struct HttpRequest {
  std::string method;
  std::string path;
  std::string body;
};

struct ChatRequest {
  std::string rendered_prompt;
  std::string model{std::string(kModelId)};
  std::size_t maximum_tokens{256};
  bool stream{false};
  bool include_usage{false};
};

struct GenerationResult {
  std::string text;
  std::size_t prompt_tokens{0};
  std::size_t completion_tokens{0};
  std::size_t proposed_drafts{0};
  std::size_t accepted_drafts{0};
  std::size_t speculative_rounds{0};
  double time_to_first_token_ms{0.0};
  double decode_ms{0.0};
  bool stopped{false};
  bool client_connected{true};
};

volatile sig_atomic_t running = 1;
volatile sig_atomic_t listener_for_signal = -1;
std::atomic<std::uint64_t> request_counter{0};

void handle_signal(int) {
  running = 0;
  if (listener_for_signal >= 0) {
    (void)::close(listener_for_signal);
    listener_for_signal = -1;
  }
}

[[nodiscard]] double monotonic_milliseconds() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration<double, std::milli>(now).count();
}

[[nodiscard]] std::int64_t unix_seconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

[[nodiscard]] std::string request_id() {
  return "chatcmpl-gfx-" + std::to_string(unix_seconds()) + "-" +
         std::to_string(request_counter.fetch_add(1));
}

[[nodiscard]] std::string json_escape(std::string_view input) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string output;
  output.reserve(input.size() + 16U);
  for (const unsigned char value : input) {
    switch (value) {
    case '\"':
      output += "\\\"";
      break;
    case '\\':
      output += "\\\\";
      break;
    case '\b':
      output += "\\b";
      break;
    case '\f':
      output += "\\f";
      break;
    case '\n':
      output += "\\n";
      break;
    case '\r':
      output += "\\r";
      break;
    case '\t':
      output += "\\t";
      break;
    default:
      if (value < 0x20U) {
        output += "\\u00";
        output.push_back(hex[value >> 4U]);
        output.push_back(hex[value & 0x0fU]);
      } else {
        output.push_back(static_cast<char>(value));
      }
    }
  }
  return output;
}

[[nodiscard]] std::string
error_json(std::string_view message,
           std::string_view code = "invalid_request") {
  return "{\"error\":{\"message\":\"" + json_escape(message) +
         "\",\"type\":\"invalid_request_error\",\"param\":null,\"code\":\"" +
         json_escape(code) + "\"}}";
}

[[nodiscard]] bool send_all(int descriptor, std::string_view data) {
  std::size_t offset = 0;
  while (offset < data.size()) {
    const auto written = ::send(descriptor, data.data() + offset,
                                data.size() - offset, MSG_NOSIGNAL);
    if (written > 0) {
      offset += static_cast<std::size_t>(written);
      continue;
    }
    if (written < 0 && errno == EINTR)
      continue;
    return false;
  }
  return true;
}

[[nodiscard]] bool send_response(int descriptor, int status,
                                 std::string_view reason,
                                 std::string_view content_type,
                                 std::string_view body) {
  const std::string header =
      "HTTP/1.1 " + std::to_string(status) + " " + std::string(reason) +
      "\r\nContent-Type: " + std::string(content_type) +
      "\r\nContent-Length: " + std::to_string(body.size()) +
      "\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n\r\n";
  return send_all(descriptor, header) && send_all(descriptor, body);
}

[[nodiscard]] bool send_json(int descriptor, int status,
                             std::string_view reason, std::string_view body) {
  return send_response(descriptor, status, reason,
                       "application/json; charset=utf-8", body);
}

[[nodiscard]] bool send_sse_headers(int descriptor) {
  return send_all(descriptor,
                  "HTTP/1.1 200 OK\r\n"
                  "Content-Type: text/event-stream; charset=utf-8\r\n"
                  "Cache-Control: no-cache\r\n"
                  "Connection: close\r\n"
                  "X-Accel-Buffering: no\r\n"
                  "X-Content-Type-Options: nosniff\r\n\r\n");
}

[[nodiscard]] bool send_sse_data(int descriptor, std::string_view json) {
  return send_all(descriptor, "data: ") && send_all(descriptor, json) &&
         send_all(descriptor, "\n\n");
}

[[nodiscard]] std::string lowercase(std::string_view input) {
  std::string output(input);
  std::transform(output.begin(), output.end(), output.begin(), [](char value) {
    if (value >= 'A' && value <= 'Z')
      return static_cast<char>(value + ('a' - 'A'));
    return value;
  });
  return output;
}

[[nodiscard]] bool parse_size(std::string_view text, std::size_t &value) {
  if (text.empty())
    return false;
  std::uint64_t parsed = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  value = static_cast<std::size_t>(parsed);
  return true;
}

[[nodiscard]] bool read_http_request(int descriptor, HttpRequest &request,
                                     std::string &error) {
  std::string bytes;
  bytes.reserve(4096);
  std::size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    char buffer[4096];
    const auto count = ::recv(descriptor, buffer, sizeof(buffer), 0);
    if (count == 0) {
      error = "connection closed before the HTTP headers completed";
      return false;
    }
    if (count < 0) {
      if (errno == EINTR)
        continue;
      error = "cannot read HTTP request";
      return false;
    }
    bytes.append(buffer, static_cast<std::size_t>(count));
    if (bytes.size() > kMaximumHeaderBytes) {
      error = "HTTP headers exceed 64 KiB";
      return false;
    }
    header_end = bytes.find("\r\n\r\n");
  }

  const auto line_end = bytes.find("\r\n");
  if (line_end == std::string::npos) {
    error = "malformed HTTP request line";
    return false;
  }
  const std::string_view line(bytes.data(), line_end);
  const auto first_space = line.find(' ');
  const auto second_space = first_space == std::string_view::npos
                                ? std::string_view::npos
                                : line.find(' ', first_space + 1U);
  if (first_space == std::string_view::npos ||
      second_space == std::string_view::npos) {
    error = "malformed HTTP request line";
    return false;
  }
  request.method = std::string(line.substr(0, first_space));
  request.path = std::string(
      line.substr(first_space + 1U, second_space - first_space - 1U));
  if (const auto query = request.path.find('?'); query != std::string::npos) {
    request.path.resize(query);
  }

  std::size_t content_length = 0;
  std::size_t cursor = line_end + 2U;
  while (cursor < header_end) {
    const auto end = bytes.find("\r\n", cursor);
    if (end == std::string::npos || end > header_end)
      break;
    const std::string_view header(bytes.data() + cursor, end - cursor);
    const auto colon = header.find(':');
    if (colon == std::string_view::npos) {
      error = "malformed HTTP header";
      return false;
    }
    const auto name = lowercase(header.substr(0, colon));
    auto value = header.substr(colon + 1U);
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
      value.remove_prefix(1U);
    }
    if (name == "content-length" && !parse_size(value, content_length)) {
      error = "invalid Content-Length header";
      return false;
    }
    if (name == "transfer-encoding" && lowercase(value) != "identity") {
      error = "chunked request bodies are not supported";
      return false;
    }
    cursor = end + 2U;
  }
  if (content_length > kMaximumBodyBytes) {
    error = "HTTP body exceeds 1 MiB";
    return false;
  }

  const std::size_t body_start = header_end + 4U;
  while (bytes.size() - body_start < content_length) {
    char buffer[8192];
    const auto needed =
        std::min(sizeof(buffer), content_length - (bytes.size() - body_start));
    const auto count = ::recv(descriptor, buffer, needed, 0);
    if (count == 0) {
      error = "connection closed before the HTTP body completed";
      return false;
    }
    if (count < 0) {
      if (errno == EINTR)
        continue;
      error = "cannot read HTTP request body";
      return false;
    }
    bytes.append(buffer, static_cast<std::size_t>(count));
  }
  request.body.assign(bytes.data() + body_start, content_length);
  return true;
}

[[nodiscard]] bool optional_bool(simdjson::dom::element root,
                                 std::string_view name, bool &value,
                                 std::string &error) {
  const auto field = root[name];
  if (field.error() == simdjson::NO_SUCH_FIELD)
    return true;
  if (field.error() || field.get(value)) {
    error = std::string(name) + " must be a boolean";
    return false;
  }
  return true;
}

[[nodiscard]] bool optional_positive_integer(simdjson::dom::element root,
                                             std::string_view name,
                                             std::size_t &value,
                                             std::string &error) {
  const auto field = root[name];
  if (field.error() == simdjson::NO_SUCH_FIELD)
    return true;
  std::uint64_t parsed = 0;
  if (field.error() || field.get(parsed) || parsed == 0U ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    error = std::string(name) + " must be a positive integer";
    return false;
  }
  value = static_cast<std::size_t>(parsed);
  return true;
}

[[nodiscard]] bool parse_chat_request(std::string_view body,
                                      const ServerOptions &options,
                                      ChatRequest &request,
                                      std::string &error) {
  simdjson::dom::parser parser;
  simdjson::padded_string padded(body);
  simdjson::dom::element root;
  if (const auto parse_error = parser.parse(padded).get(root); parse_error) {
    error = "request body is not valid JSON";
    return false;
  }

  const auto model_field = root["model"];
  if (!model_field.error()) {
    std::string_view model;
    if (model_field.get(model)) {
      error = "model must be a string";
      return false;
    }
    if (model != kModelId) {
      error = "unknown model; use " + std::string(kModelId);
      return false;
    }
    request.model = std::string(model);
  }
  if (!optional_positive_integer(root, "max_tokens", request.maximum_tokens,
                                 error)) {
    return false;
  }
  if (root["max_tokens"].error() == simdjson::NO_SUCH_FIELD &&
      !optional_positive_integer(root, "max_completion_tokens",
                                 request.maximum_tokens, error)) {
    return false;
  }
  if (request.maximum_tokens > options.maximum_output_tokens) {
    error = "requested output exceeds the server maximum of " +
            std::to_string(options.maximum_output_tokens) + " tokens";
    return false;
  }
  if (!optional_bool(root, "stream", request.stream, error))
    return false;

  auto stream_options = root["stream_options"];
  if (!stream_options.error()) {
    if (!request.stream) {
      error = "stream_options requires stream=true";
      return false;
    }
    if (!optional_bool(stream_options.value(), "include_usage",
                       request.include_usage, error)) {
      return false;
    }
  }

  const auto temperature_field = root["temperature"];
  if (!temperature_field.error()) {
    double temperature = 0.0;
    if (temperature_field.get(temperature) || temperature != 0.0) {
      error =
          "this release supports greedy decoding only; set temperature to 0";
      return false;
    }
  }
  const auto n_field = root["n"];
  if (!n_field.error()) {
    std::uint64_t n = 0;
    if (n_field.get(n) || n != 1U) {
      error = "this release supports n=1 only";
      return false;
    }
  }
  if (!root["tools"].error() || !root["response_format"].error() ||
      !root["stop"].error()) {
    error =
        "tools, response_format, and custom stop sequences are not implemented";
    return false;
  }

  simdjson::dom::array messages;
  if (const auto message_error = root["messages"].get(messages);
      message_error || messages.size() == 0U) {
    error = "messages must be a non-empty array";
    return false;
  }
  std::string rendered;
  for (const auto message : messages) {
    std::string_view role;
    std::string_view content;
    if (message["role"].get(role) || message["content"].get(content)) {
      error = "every message must contain string role and content fields";
      return false;
    }
    if (role == "developer")
      role = "system";
    if (role != "system" && role != "user" && role != "assistant") {
      error =
          "supported message roles are developer, system, user, and assistant";
      return false;
    }
    rendered += "<|im_start|>";
    rendered.append(role);
    rendered += '\n';
    rendered.append(content);
    rendered += "<|im_end|>\n";
  }
  rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  request.rendered_prompt = std::move(rendered);
  return true;
}

[[nodiscard]] bool terminal_token(std::uint32_t token) {
  return token == kEndToken || token == kEndOfTextToken;
}

[[nodiscard]] std::size_t valid_utf8_prefix(std::string_view text) {
  std::size_t offset = 0;
  while (offset < text.size()) {
    const auto first = static_cast<unsigned char>(text[offset]);
    std::size_t length = 0;
    if (first < 0x80U)
      length = 1;
    else if ((first & 0xe0U) == 0xc0U)
      length = 2;
    else if ((first & 0xf0U) == 0xe0U)
      length = 3;
    else if ((first & 0xf8U) == 0xf0U)
      length = 4;
    else
      return offset;
    if (offset + length > text.size())
      return offset;
    for (std::size_t index = 1; index < length; ++index) {
      if ((static_cast<unsigned char>(text[offset + index]) & 0xc0U) != 0x80U) {
        return offset;
      }
    }
    offset += length;
  }
  return offset;
}

using TextCallback = std::function<bool(std::string_view)>;

[[nodiscard]] gfxinfer::Result<GenerationResult>
generate(gfxinfer::Engine &engine, const gfxinfer::QwenTokenizer &tokenizer,
         const ServerOptions &options, const ChatRequest &request,
         const TextCallback &on_text) {
  auto prompt_tokens = tokenizer.encode(request.rendered_prompt);
  if (!prompt_tokens.ok())
    return prompt_tokens.status();
  if (prompt_tokens.value().empty() ||
      prompt_tokens.value().size() + request.maximum_tokens > options.context) {
    return gfxinfer::Status{
        gfxinfer::ErrorCode::invalid_argument,
        "rendered messages and requested output exceed the server context"};
  }

  auto session = gfxinfer::QwenDecodeSession::create(
      engine, {.maximum_context_tokens = options.context,
               .activation_mode = options.activation,
               .graph_verifier_batch = static_cast<unsigned>(
                   options.draft_count == 0U ? 0U : options.draft_count + 1U)});
  if (!session.ok())
    return session.status();

  GenerationResult output;
  output.prompt_tokens = prompt_tokens.value().size();
  const double prefill_begin = monotonic_milliseconds();
  auto first = session.value().greedy_step(prompt_tokens.value().front());
  if (!first.ok())
    return first.status();
  std::uint32_t next = first.value();
  for (std::size_t index = 1; index < prompt_tokens.value().size(); ++index) {
    const auto token = prompt_tokens.value()[index];
    if (options.draft_count != 0U) {
      auto sync = session.value().mtp_draft(token);
      if (!sync.ok())
        return sync.status();
    }
    auto target = session.value().greedy_step(token);
    if (!target.ok())
      return target.status();
    next = target.value();
  }
  const double first_token_ready = monotonic_milliseconds();
  output.time_to_first_token_ms = first_token_ready - prefill_begin;

  std::vector<std::uint32_t> content_tokens;
  content_tokens.reserve(request.maximum_tokens);
  std::size_t sent_bytes = 0;
  auto emit_content = [&]() -> gfxinfer::Status {
    auto decoded = tokenizer.decode(content_tokens);
    if (!decoded.ok())
      return decoded.status();
    output.text = std::move(decoded.value());
    const auto valid_bytes = valid_utf8_prefix(output.text);
    if (valid_bytes > sent_bytes && on_text) {
      if (!on_text(std::string_view(output.text)
                       .substr(sent_bytes, valid_bytes - sent_bytes))) {
        output.client_connected = false;
        return gfxinfer::Status::ok_status();
      }
      sent_bytes = valid_bytes;
    }
    return gfxinfer::Status::ok_status();
  };

  auto consume_token = [&](std::uint32_t token) -> gfxinfer::Status {
    next = token;
    ++output.completion_tokens;
    if (terminal_token(token)) {
      output.stopped = true;
      return gfxinfer::Status::ok_status();
    }
    content_tokens.push_back(token);
    return emit_content();
  };

  auto consume_status = consume_token(next);
  if (!consume_status.ok())
    return consume_status;
  while (output.completion_tokens < request.maximum_tokens &&
         output.client_connected && !output.stopped) {
    const auto remaining = request.maximum_tokens - output.completion_tokens;
    if (options.draft_count == 0U || remaining == 1U) {
      auto target = session.value().greedy_step(next);
      if (!target.ok())
        return target.status();
      consume_status = consume_token(target.value());
      if (!consume_status.ok())
        return consume_status;
      continue;
    }

    const auto round_depth = std::min(options.draft_count, remaining - 1U);
    auto speculative = session.value().speculative_step(next, round_depth);
    if (!speculative.ok())
      return speculative.status();
    output.proposed_drafts += speculative.value().proposed_drafts;
    output.accepted_drafts += speculative.value().accepted_drafts;
    ++output.speculative_rounds;
    for (const auto token : speculative.value().tokens) {
      consume_status = consume_token(token);
      if (!consume_status.ok())
        return consume_status;
      if (!output.client_connected || output.stopped ||
          output.completion_tokens == request.maximum_tokens) {
        break;
      }
    }
  }
  output.decode_ms = monotonic_milliseconds() - first_token_ready;
  return output;
}

[[nodiscard]] std::string completion_json(const std::string &id,
                                          std::int64_t created,
                                          const ChatRequest &request,
                                          const GenerationResult &result) {
  const std::string finish_reason = result.stopped ? "stop" : "length";
  return "{\"id\":\"" + json_escape(id) +
         "\",\"object\":\"chat.completion\",\"created\":" +
         std::to_string(created) + ",\"model\":\"" +
         json_escape(request.model) +
         "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
         "\"content\":\"" +
         json_escape(result.text) +
         "\",\"refusal\":null},\"logprobs\":null,\"finish_reason\":\"" +
         finish_reason + "\"}],\"usage\":{\"prompt_tokens\":" +
         std::to_string(result.prompt_tokens) +
         ",\"completion_tokens\":" + std::to_string(result.completion_tokens) +
         ",\"total_tokens\":" +
         std::to_string(result.prompt_tokens + result.completion_tokens) + "}}";
}

[[nodiscard]] std::string
stream_chunk(const std::string &id, std::int64_t created,
             const ChatRequest &request, std::string_view delta,
             bool include_role, std::string_view finish_reason = {}) {
  std::string delta_json = "{";
  if (include_role)
    delta_json += "\"role\":\"assistant\"";
  if (!delta.empty()) {
    if (include_role)
      delta_json += ',';
    delta_json += "\"content\":\"" + json_escape(delta) + "\"";
  }
  delta_json += '}';
  return "{\"id\":\"" + json_escape(id) +
         "\",\"object\":\"chat.completion.chunk\",\"created\":" +
         std::to_string(created) + ",\"model\":\"" +
         json_escape(request.model) +
         "\",\"choices\":[{\"index\":0,\"delta\":" + delta_json +
         ",\"logprobs\":null,\"finish_reason\":" +
         (finish_reason.empty() ? std::string("null")
                                : "\"" + std::string(finish_reason) + "\"") +
         "}],\"usage\":null}";
}

void log_generation(const std::string &id, const GenerationResult &result) {
  const double speed =
      result.completion_tokens > 1U && result.decode_ms > 0.0
          ? static_cast<double>(result.completion_tokens - 1U) * 1000.0 /
                result.decode_ms
          : 0.0;
  std::cerr << id << " prompt=" << result.prompt_tokens
            << " completion=" << result.completion_tokens
            << " ttft_ms=" << result.time_to_first_token_ms
            << " decode_tps=" << speed;
  if (result.proposed_drafts != 0U) {
    std::cerr << " mtp_acceptance="
              << static_cast<double>(result.accepted_drafts) * 100.0 /
                     static_cast<double>(result.proposed_drafts)
              << '%';
  }
  std::cerr << '\n';
}

void handle_chat(int descriptor, gfxinfer::Engine &engine,
                 const gfxinfer::QwenTokenizer &tokenizer,
                 const ServerOptions &options, const HttpRequest &http) {
  ChatRequest request;
  std::string parse_error;
  if (!parse_chat_request(http.body, options, request, parse_error)) {
    (void)send_json(descriptor, 400, "Bad Request", error_json(parse_error));
    return;
  }

  const auto id = request_id();
  const auto created = unix_seconds();
  if (!request.stream) {
    auto result = generate(engine, tokenizer, options, request, {});
    if (!result.ok()) {
      (void)send_json(descriptor, 400, "Bad Request",
                      error_json(result.status().message()));
      return;
    }
    log_generation(id, result.value());
    (void)send_json(descriptor, 200, "OK",
                    completion_json(id, created, request, result.value()));
    return;
  }

  if (!send_sse_headers(descriptor) ||
      !send_sse_data(descriptor,
                     stream_chunk(id, created, request, {}, true))) {
    return;
  }
  auto result =
      generate(engine, tokenizer, options, request, [&](std::string_view text) {
        return send_sse_data(descriptor,
                             stream_chunk(id, created, request, text, false));
      });
  if (!result.ok()) {
    (void)send_sse_data(
        descriptor, error_json(result.status().message(), "generation_error"));
    (void)send_all(descriptor, "data: [DONE]\n\n");
    return;
  }
  if (!result.value().client_connected)
    return;
  log_generation(id, result.value());
  const std::string finish_reason = result.value().stopped ? "stop" : "length";
  if (!send_sse_data(descriptor, stream_chunk(id, created, request, {}, false,
                                              finish_reason))) {
    return;
  }
  if (request.include_usage) {
    const std::string usage =
        "{\"id\":\"" + json_escape(id) +
        "\",\"object\":\"chat.completion.chunk\",\"created\":" +
        std::to_string(created) + ",\"model\":\"" + json_escape(request.model) +
        "\",\"choices\":[],\"usage\":{\"prompt_tokens\":" +
        std::to_string(result.value().prompt_tokens) +
        ",\"completion_tokens\":" +
        std::to_string(result.value().completion_tokens) +
        ",\"total_tokens\":" +
        std::to_string(result.value().prompt_tokens +
                       result.value().completion_tokens) +
        "}}";
    if (!send_sse_data(descriptor, usage))
      return;
  }
  (void)send_all(descriptor, "data: [DONE]\n\n");
}

void handle_connection(int descriptor, gfxinfer::Engine &engine,
                       const gfxinfer::QwenTokenizer &tokenizer,
                       const ServerOptions &options) {
  timeval timeout{300, 0};
  (void)setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                   sizeof(timeout));
  (void)setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                   sizeof(timeout));

  HttpRequest request;
  std::string error;
  if (!read_http_request(descriptor, request, error)) {
    (void)send_json(descriptor, 400, "Bad Request", error_json(error));
    return;
  }
  if (request.method == "GET" && request.path == "/health") {
    (void)send_json(descriptor, 200, "OK", "{\"status\":\"ok\"}");
    return;
  }
  if (request.method == "GET" && request.path == "/v1/models") {
    const std::string body =
        "{\"object\":\"list\",\"data\":[{\"id\":\"" + std::string(kModelId) +
        "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"local\"}]}";
    (void)send_json(descriptor, 200, "OK", body);
    return;
  }
  if (request.method == "POST" && request.path == "/v1/chat/completions") {
    handle_chat(descriptor, engine, tokenizer, options, request);
    return;
  }
  (void)send_json(descriptor, 404, "Not Found",
                  error_json("route not found", "not_found"));
}

[[nodiscard]] bool parse_activation(std::string_view value,
                                    gfxinfer::ActivationMode &output) {
  if (value == "f16")
    output = gfxinfer::ActivationMode::fp16;
  else if (value == "a8")
    output = gfxinfer::ActivationMode::int8;
  else if (value == "a4")
    output = gfxinfer::ActivationMode::int4;
  else if (value == "a4w2")
    output = gfxinfer::ActivationMode::int4_w2;
  else if (value == "a4w4")
    output = gfxinfer::ActivationMode::int4_w4;
  else
    return false;
  return true;
}

void print_usage() {
  std::cout
      << "GFXInfer server " << gfxinfer::kVersion << "\n\n"
      << "Usage:\n"
      << "  gfxinfer-server <artifact.gfxi> [--host 127.0.0.1] [--port 8000]\n"
      << "      [--context 4096] [--max-output-tokens 1024] [--drafts 0..31]\n"
      << "      [--activation f16|a8|a4|a4w2|a4w4]\n";
}

[[nodiscard]] bool parse_options(int argc, char **argv,
                                 ServerOptions &options) {
  if (argc < 2 || ((argc - 2) % 2) != 0)
    return false;
  options.artifact = argv[1];
  for (int index = 2; index < argc; index += 2) {
    const std::string_view name = argv[index];
    const std::string_view value = argv[index + 1];
    std::size_t parsed = 0;
    if (name == "--host") {
      options.host = value;
    } else if (name == "--activation") {
      if (!parse_activation(value, options.activation))
        return false;
    } else if (name == "--port") {
      if (!parse_size(value, parsed) || parsed == 0U || parsed > 65535U)
        return false;
      options.port = static_cast<std::uint16_t>(parsed);
    } else if (name == "--context") {
      if (!parse_size(value, options.context) || options.context == 0U ||
          options.context > gfxinfer::Qwen38Spec::maximum_context_tokens) {
        return false;
      }
    } else if (name == "--max-output-tokens") {
      if (!parse_size(value, options.maximum_output_tokens) ||
          options.maximum_output_tokens == 0U) {
        return false;
      }
    } else if (name == "--drafts") {
      if (!parse_size(value, options.draft_count) ||
          options.draft_count > 31U) {
        return false;
      }
    } else {
      return false;
    }
  }
  return options.maximum_output_tokens <= options.context;
}

[[nodiscard]] int listen_socket(const ServerOptions &options) {
  const int descriptor = ::socket(AF_INET, SOCK_STREAM, 0);
  if (descriptor < 0)
    return -1;
  const int enabled = 1;
  (void)setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &enabled,
                   sizeof(enabled));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(options.port);
  if (inet_pton(AF_INET, options.host.c_str(), &address.sin_addr) != 1 ||
      ::bind(descriptor, reinterpret_cast<const sockaddr *>(&address),
             sizeof(address)) != 0 ||
      ::listen(descriptor, 16) != 0) {
    (void)::close(descriptor);
    return -1;
  }
  return descriptor;
}

} // namespace

int main(int argc, char **argv) {
  ServerOptions options;
  if (!parse_options(argc, argv, options)) {
    print_usage();
    return 1;
  }
  (void)::signal(SIGINT, handle_signal);
  (void)::signal(SIGTERM, handle_signal);
  (void)::signal(SIGPIPE, SIG_IGN);

  gfxinfer::EngineConfig engine_config;
  engine_config.artifact = options.artifact;
  auto engine = gfxinfer::Engine::create(std::move(engine_config));
  if (!engine.ok()) {
    std::cerr << "error: " << engine.status().message() << '\n';
    return 1;
  }
  auto tokenizer = gfxinfer::QwenTokenizer::from_json(
      engine.value().resource("resources/tokenizer.json"));
  if (!tokenizer.ok()) {
    std::cerr << "error: " << tokenizer.status().message() << '\n';
    return 1;
  }
  const int listener = listen_socket(options);
  if (listener < 0) {
    std::cerr << "error: cannot listen on " << options.host << ':'
              << options.port << ": " << std::strerror(errno) << '\n';
    return 1;
  }
  std::cout << "GFXInfer ready\n"
            << "  model:    " << kModelId << '\n'
            << "  endpoint: http://" << options.host << ':' << options.port
            << "/v1/chat/completions\n"
            << "  policy:   greedy A4, MTP depth " << options.draft_count
            << '\n'
            << "  serving:  one request at a time\n"
            << std::flush;
  if (options.host != "127.0.0.1") {
    std::cerr << "warning: non-loopback serving has no authentication or TLS\n";
  }

  listener_for_signal = listener;
  while (running != 0) {
    sockaddr_in client_address{};
    socklen_t address_size = sizeof(client_address);
    const int client = ::accept(
        listener, reinterpret_cast<sockaddr *>(&client_address), &address_size);
    if (client < 0) {
      if (running == 0)
        break;
      if (errno == EINTR)
        continue;
      std::cerr << "error: accept failed: " << std::strerror(errno) << '\n';
      break;
    }
    handle_connection(client, engine.value(), tokenizer.value(), options);
    (void)::close(client);
  }
  if (listener_for_signal >= 0) {
    (void)::close(listener);
    listener_for_signal = -1;
  }
  return 0;
}
