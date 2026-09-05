// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace gfxinfer {

enum class ErrorCode {
  ok = 0,
  invalid_argument,
  io_error,
  corrupt_artifact,
  unsupported,
  hip_error,
  out_of_memory,
  internal,
};

class Status {
 public:
  Status() = default;
  Status(ErrorCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  [[nodiscard]] static Status ok_status() { return {}; }
  [[nodiscard]] bool ok() const noexcept { return code_ == ErrorCode::ok; }
  [[nodiscard]] ErrorCode code() const noexcept { return code_; }
  [[nodiscard]] const std::string& message() const noexcept { return message_; }

 private:
  ErrorCode code_{ErrorCode::ok};
  std::string message_;
};

template <typename T>
class Result {
 public:
  Result(T value) : value_(std::move(value)) {}
  Result(Status status) : value_(std::move(status)) {
    if (std::get<Status>(value_).ok()) {
      throw std::invalid_argument("Result error must not contain OK status");
    }
  }

  [[nodiscard]] bool ok() const noexcept { return std::holds_alternative<T>(value_); }
  [[nodiscard]] T& value() & { return std::get<T>(value_); }
  [[nodiscard]] const T& value() const& { return std::get<T>(value_); }
  [[nodiscard]] T&& value() && { return std::get<T>(std::move(value_)); }
  [[nodiscard]] const Status& status() const { return std::get<Status>(value_); }

 private:
  std::variant<T, Status> value_;
};

}  // namespace gfxinfer
