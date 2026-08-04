#pragma once

#include <infiniband/verbs.h>

#include <cstddef>
#include <vector>

#include "micro_ccl/verbs/device.hpp"

namespace micro_ccl {

// Every ibv_post_send/ibv_post_recv is fire-and-forget from the caller's
// point of view -- the function returns as soon as the work request is
// queued, before the NIC has actually moved any bytes. The completion
// queue is how you find out what actually happened: the NIC pushes a
// completion queue entry (CQE) here once a work request finishes,
// successfully or not. poll() is a non-blocking drain of whatever CQEs are
// ready right now; callers that need to block until N completions arrive
// wrap poll() in their own retry loop (see transport/), because "block vs.
// spin vs. event-driven" is a data-path policy decision, not something this
// thin wrapper should hardcode.
class CompletionQueue {
 public:
  explicit CompletionQueue(Device& device, int capacity = 128);
  ~CompletionQueue();

  CompletionQueue(const CompletionQueue&) = delete;
  CompletionQueue& operator=(const CompletionQueue&) = delete;
  CompletionQueue(CompletionQueue&& other) noexcept;
  CompletionQueue& operator=(CompletionQueue&& other) noexcept;

  // Drains up to max_entries completions into `out` (cleared first) and
  // returns how many were found. Never blocks.
  int poll(std::vector<ibv_wc>& out, int max_entries = 16);

  ibv_cq* native() const { return cq_; }

 private:
  ibv_cq* cq_ = nullptr;
};

}  // namespace micro_ccl
