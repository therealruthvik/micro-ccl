#include "micro_ccl/verbs/memory_region.hpp"

#include "micro_ccl/detail/error.hpp"

namespace micro_ccl {

MemoryRegion::MemoryRegion(ProtectionDomain& pd, void* addr, size_t length,
                            int access)
    : mr_(check_ptr(ibv_reg_mr(pd.native(), addr, length, access),
                     "ibv_reg_mr")) {}

MemoryRegion::~MemoryRegion() {
  if (mr_ != nullptr) {
    ibv_dereg_mr(mr_);
  }
}

MemoryRegion::MemoryRegion(MemoryRegion&& other) noexcept : mr_(other.mr_) {
  other.mr_ = nullptr;
}

MemoryRegion& MemoryRegion::operator=(MemoryRegion&& other) noexcept {
  if (this != &other) {
    if (mr_ != nullptr) {
      ibv_dereg_mr(mr_);
    }
    mr_ = other.mr_;
    other.mr_ = nullptr;
  }
  return *this;
}

}  // namespace micro_ccl
