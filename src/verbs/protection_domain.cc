#include "micro_ccl/verbs/protection_domain.hpp"

#include "micro_ccl/detail/error.hpp"

namespace micro_ccl {

ProtectionDomain::ProtectionDomain(Device& device)
    : pd_(check_ptr(ibv_alloc_pd(device.context()), "ibv_alloc_pd")) {}

ProtectionDomain::~ProtectionDomain() {
  if (pd_ != nullptr) {
    ibv_dealloc_pd(pd_);
  }
}

ProtectionDomain::ProtectionDomain(ProtectionDomain&& other) noexcept
    : pd_(other.pd_) {
  other.pd_ = nullptr;
}

ProtectionDomain& ProtectionDomain::operator=(
    ProtectionDomain&& other) noexcept {
  if (this != &other) {
    if (pd_ != nullptr) {
      ibv_dealloc_pd(pd_);
    }
    pd_ = other.pd_;
    other.pd_ = nullptr;
  }
  return *this;
}

}  // namespace micro_ccl
