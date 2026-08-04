#pragma once

#include <infiniband/verbs.h>

#include <cstddef>

#include "micro_ccl/verbs/protection_domain.hpp"

namespace micro_ccl {

// Registers an existing block of memory with the NIC so it can be used
// directly as a DMA source/destination -- this is the object that makes
// zero-copy possible. Registration (ibv_reg_mr) pins the pages (so the
// kernel can never swap them out from under an in-flight DMA) and returns
// two keys: lkey, which *this* process uses locally when posting sends/
// recvs against the buffer, and rkey, which we hand to a *peer* so it can
// target this memory directly with an RDMA read/write, without our CPU
// touching the data at all.
//
// micro-ccl registers each application buffer exactly once up front and
// reuses it for the life of the communicator -- registration is not free
// (it's a syscall that pins pages and programs the NIC's MMU), so doing it
// per-message would defeat the point of zero-copy.
//
// This type deliberately does not own the underlying memory (no malloc/free
// here) -- it wraps a pointer the caller already owns, matching how
// std::span or std::string_view work: a non-owning view over someone else's
// storage, with its own separate lifetime concern (registration) layered on
// top.
class MemoryRegion {
 public:
  // access defaults to a superset (local write + remote read + remote
  // write) so the same registered buffer works whether a collective uses
  // two-sided send/recv (only needs lkey) or one-sided RDMA read/write
  // (needs rkey too). Requesting capabilities you don't end up using costs
  // nothing at the verbs layer.
  MemoryRegion(ProtectionDomain& pd, void* addr, size_t length,
               int access = IBV_ACCESS_LOCAL_WRITE |
                            IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_READ);
  ~MemoryRegion();

  MemoryRegion(const MemoryRegion&) = delete;
  MemoryRegion& operator=(const MemoryRegion&) = delete;
  MemoryRegion(MemoryRegion&& other) noexcept;
  MemoryRegion& operator=(MemoryRegion&& other) noexcept;

  uint32_t lkey() const { return mr_->lkey; }
  uint32_t rkey() const { return mr_->rkey; }
  void* addr() const { return mr_->addr; }
  size_t length() const { return mr_->length; }

  ibv_mr* native() const { return mr_; }

 private:
  ibv_mr* mr_ = nullptr;
};

}  // namespace micro_ccl
