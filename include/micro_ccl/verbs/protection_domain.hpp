#pragma once

#include <infiniband/verbs.h>

#include "micro_ccl/verbs/device.hpp"

namespace micro_ccl {

// A protection domain is an access-control grouping, not a resource with
// real weight -- it costs nothing to create and holds no buffers. Its only
// job: a QueuePair and a MemoryRegion can reference each other (e.g. a send
// WR's lkey) only if they were both created from the same ProtectionDomain.
// One PD per process is enough for this library; we still model it as its
// own RAII type because ibv_pd* is a kernel handle like any other and
// should not leak into application code unwrapped.
class ProtectionDomain {
 public:
  explicit ProtectionDomain(Device& device);
  ~ProtectionDomain();

  ProtectionDomain(const ProtectionDomain&) = delete;
  ProtectionDomain& operator=(const ProtectionDomain&) = delete;
  ProtectionDomain(ProtectionDomain&& other) noexcept;
  ProtectionDomain& operator=(ProtectionDomain&& other) noexcept;

  ibv_pd* native() const { return pd_; }

 private:
  ibv_pd* pd_ = nullptr;
};

}  // namespace micro_ccl
