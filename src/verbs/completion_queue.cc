#include "micro_ccl/verbs/completion_queue.hpp"

#include <stdexcept>
#include <string>

#include "micro_ccl/detail/error.hpp"

namespace micro_ccl {

CompletionQueue::CompletionQueue(Device& device, int capacity)
    : cq_(check_ptr(
          ibv_create_cq(device.context(), capacity, /*cq_context=*/nullptr,
                         /*channel=*/nullptr, /*comp_vector=*/0),
          "ibv_create_cq")) {}

CompletionQueue::~CompletionQueue() {
  if (cq_ != nullptr) {
    ibv_destroy_cq(cq_);
  }
}

CompletionQueue::CompletionQueue(CompletionQueue&& other) noexcept
    : cq_(other.cq_) {
  other.cq_ = nullptr;
}

CompletionQueue& CompletionQueue::operator=(CompletionQueue&& other) noexcept {
  if (this != &other) {
    if (cq_ != nullptr) {
      ibv_destroy_cq(cq_);
    }
    cq_ = other.cq_;
    other.cq_ = nullptr;
  }
  return *this;
}

int CompletionQueue::poll(std::vector<ibv_wc>& out, int max_entries) {
  out.clear();
  out.resize(max_entries);
  int n = ibv_poll_cq(cq_, max_entries, out.data());
  // ibv_poll_cq returns a negative value only on a genuine driver-level
  // error (not "zero completions ready", which is the normal n == 0 case).
  // Unlike ibv_modify_qp et al., the verbs spec does not guarantee the
  // magnitude is a meaningful errno, so we surface it as a plain error
  // rather than pretending it decodes to a strerror() string.
  if (n < 0) {
    throw std::runtime_error("ibv_poll_cq failed, rc=" + std::to_string(n));
  }
  out.resize(n);
  return n;
}

}  // namespace micro_ccl
