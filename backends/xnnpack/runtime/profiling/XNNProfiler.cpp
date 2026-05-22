/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/profiling/XNNProfiler.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/event_tracer.h>

// Headers used only when event tracer is compiled in.
// NOLINTBEGIN
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/types.h>

#include <cinttypes>
#include <cstring>
#include <string>
#include <unordered_map>
// NOLINTEND

namespace executorch::backends::xnnpack::delegate::profiling {

using executorch::runtime::Error;
using executorch::runtime::EventTracer;

#if defined(ET_EVENT_TRACER_ENABLED) || defined(ENABLE_XNNPACK_PROFILING)

XNNProfiler::XNNProfiler()
    : state_(XNNProfilerState::Uninitialized), run_count_(0) {}

Error XNNProfiler::initialize(xnn_runtime_t runtime) {
  runtime_ = runtime;

  // Fetch the runtime operator information from XNNPACK.
  ET_CHECK_OK_OR_RETURN_ERROR(get_runtime_num_operators());
  ET_CHECK_OK_OR_RETURN_ERROR(get_runtime_operator_names());

  state_ = XNNProfilerState::Ready;

  return Error::Ok;
}

// Two-pass query helper: probe size then fill the buffer.
static Error query_runtime_string_array(
    xnn_runtime_t runtime,
    xnn_profile_info param,
    std::vector<char>& out) {
  size_t required_size = 0;
  xnn_status status = xnn_get_runtime_profiling_info(
      runtime, param, 0, nullptr, &required_size);
  if (status == xnn_status_out_of_memory) {
    out.resize(required_size);
    status = xnn_get_runtime_profiling_info(
        runtime, param, out.size(), out.data(), &required_size);
  } else if (status == xnn_status_success && required_size == 0) {
    // Nothing to fetch (no valid operators); leave `out` empty.
    out.clear();
  }
  if (status != xnn_status_success) {
    ET_LOG(Error, "XNNPACK profiling query %d failed: %d", (int)param, status);
    return Error::Internal;
  }
  return Error::Ok;
}

Error XNNProfiler::start(EventTracer* event_tracer) {
  // Validate profiler state.
  if (state_ == XNNProfilerState::Uninitialized) {
    ET_LOG(
        Error,
        "XNNProfiler must be initialized prior to calling begin_execution.");
    return Error::InvalidState;
  } else if (state_ == XNNProfilerState::Running) {
    ET_LOG(
        Error,
        "XNNProfiler is already running. Call end_execution() before calling begin_execution().");
    return Error::InvalidState;
  }

  event_tracer_ = event_tracer;
  state_ = XNNProfilerState::Running;

  // Log the start of execution timestamp.
  start_time_ = runtime::pal_current_ticks();

  return Error::Ok;
}

Error XNNProfiler::end() {
  // Validate profiler state.
  ET_CHECK_OR_RETURN_ERROR(
      state_ == XNNProfilerState::Running,
      InvalidState,
      "XNNProfiler is not running. Ensure begin_execution() is called before end_execution().");

  // Retrieve operator timing from XNNPACK.
  ET_CHECK_OK_OR_RETURN_ERROR(get_runtime_operator_timings());

  // Best-effort: an XNNPACK that doesn't know these enums leaves them empty.
  if (get_runtime_microkernel_names() != Error::Ok) {
    microkernel_names_.clear();
  }
  if (get_runtime_microkernel_timings() != Error::Ok) {
    microkernel_timings_.clear();
  }

  if (event_tracer_ != nullptr) {
    submit_trace();
  }

  log_operator_timings();

  state_ = XNNProfilerState::Ready;
  return Error::Ok;
}

Error XNNProfiler::get_runtime_operator_names() {
  return query_runtime_string_array(
      runtime_, xnn_profile_info_operator_name, op_names_);
}

Error XNNProfiler::get_runtime_microkernel_names() {
  return query_runtime_string_array(
      runtime_, xnn_profile_info_microkernel_name, microkernel_names_);
}

Error XNNProfiler::get_runtime_num_operators() {
  size_t required_size = 0;

  xnn_status status = xnn_get_runtime_profiling_info(
      runtime_,
      xnn_profile_info_num_operators,
      sizeof(op_count_),
      &op_count_,
      &required_size);

  if (status != xnn_status_success) {
    ET_LOG(Error, "Failed to get XNNPACK operator count: %d", status);
    return Error::Internal;
  }

  return Error::Ok;
}

static Error query_runtime_uint64_array(
    xnn_runtime_t runtime,
    xnn_profile_info param,
    std::vector<uint64_t>& out,
    size_t op_count) {
  size_t required_size = 0;
  out.resize(op_count);
  xnn_status status = xnn_get_runtime_profiling_info(
      runtime,
      param,
      out.size() * sizeof(uint64_t),
      out.data(),
      &required_size);
  if (status != xnn_status_success) {
    ET_LOG(Error, "XNNPACK profiling query %d failed: %d", (int)param, status);
    return Error::Internal;
  }
  return Error::Ok;
}

Error XNNProfiler::get_runtime_operator_timings() {
  return query_runtime_uint64_array(
      runtime_, xnn_profile_info_operator_timing, op_timings_, op_count_);
}

Error XNNProfiler::get_runtime_microkernel_timings() {
  return query_runtime_uint64_array(
      runtime_,
      xnn_profile_info_microkernel_timing,
      microkernel_timings_,
      op_count_);
}

void XNNProfiler::log_operator_timings() {
#ifdef ENABLE_XNNPACK_PROFILING
  // Update running average state and log average timing for each op.
  run_count_++;
  size_t name_len = 0;
  const char* op_name = nullptr;
  auto total_time = 0.0f;

  if (op_timings_sum_.size() != op_count_) {
    op_timings_sum_ = std::vector<uint64_t>(op_count_, 0);
  }

  for (size_t i = 0; i < op_count_; i++) {
    op_name = &op_names_[name_len];
    name_len += strlen(op_name) + 1;

    op_timings_sum_[i] += op_timings_[i];
    auto avg_op_time = op_timings_sum_[i] / static_cast<float>(run_count_);
    total_time += avg_op_time;

    ET_LOG(
        Info, ">>, %s, %" PRId64 " (%f)", op_name, op_timings_[i], avg_op_time);
  }
  ET_LOG(Info, ">>, Total Time, %f", total_time);
#else
  run_count_++;
#endif
}

void XNNProfiler::submit_trace() {
  // Retrieve the system tick rate (ratio between ticks and nanoseconds).
  auto tick_ns_conv_multiplier = runtime::pal_ticks_to_ns_multiplier();

  ET_CHECK(op_timings_.size() == op_count_);
  size_t name_len = 0;
  size_t microkernel_name_len = 0;
  et_timestamp_t time = start_time_;
  std::unordered_map<std::string, uint32_t> op_counts;

  const bool have_microkernel_names = !microkernel_names_.empty();
  const bool have_microkernel_timings = microkernel_timings_.size() == op_count_;

  for (auto i = 0u; i < op_count_; i++) {
    auto op_name = &op_names_[name_len];
    name_len += strlen(op_name) + 1;

    // Walk the parallel NUL-separated microkernel-name array.
    const char* microkernel_name = "";
    if (have_microkernel_names &&
        microkernel_name_len < microkernel_names_.size()) {
      microkernel_name = &microkernel_names_[microkernel_name_len];
      microkernel_name_len += strlen(microkernel_name) + 1;
    }

    // Format the op name as {name} #{count}.
    auto op_name_str = std::string(op_name);
    op_counts[op_name_str]++;
    auto name_formatted =
        op_name_str + " #" + std::to_string(op_counts[op_name_str]);

    // Convert from microseconds (XNNPACK) to PAL ticks (ET).
    // The tick_ns_conv_ratio is ns / tick. We want ticks:
    //  ticks = us * (ns / us) / conv_ratio
    //        = us * 1000 * conv_ratio.denom / conv_ratio.num
    auto interval_ticks = static_cast<et_timestamp_t>(
        op_timings_[i] * 1000 * tick_ns_conv_multiplier.denominator /
        tick_ns_conv_multiplier.numerator);

    auto end_time = time + interval_ticks;

    // Pack metadata as <symbol>\0<microkernel us> for etdump_summary.py.
    std::string metadata;
    if (microkernel_name[0] != '\0' ||
        (have_microkernel_timings && microkernel_timings_[i] != 0)) {
      metadata.assign(microkernel_name);
      metadata.push_back('\0');
      if (have_microkernel_timings) {
        metadata.append(std::to_string(microkernel_timings_[i]));
      }
    }
    executorch::runtime::event_tracer_log_profiling_delegate(
        event_tracer_,
        name_formatted.c_str(),
        /*delegate_debug_id=*/static_cast<executorch::runtime::DebugHandle>(-1),
        time,
        end_time,
        metadata.empty() ? nullptr : metadata.data(),
        metadata.size());

    // Assume that the next op starts immediately after the previous op.
    // This may not be strictly true, but it should be close enough.
    // Ideally, we'll get the start and end times from XNNPACK in the
    // future.
    time = end_time;
  }
}

#else // defined(ET_EVENT_TRACER_ENABLED) || defined(ENABLE_XNNPACK_PROFILING)

// Stub implementation for when profiling is disabled.
XNNProfiler::XNNProfiler() {}

Error XNNProfiler::initialize(xnn_runtime_t runtime) {
  (void)runtime;
  return Error::Ok;
}

Error XNNProfiler::start(EventTracer* event_tracer) {
  (void)event_tracer;
  return Error::Ok;
}

Error XNNProfiler::end() {
  return Error::Ok;
}

#endif

} // namespace executorch::backends::xnnpack::delegate::profiling
