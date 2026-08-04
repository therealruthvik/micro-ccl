#pragma once

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

namespace micro_ccl {

// All verbs failures surface as this, with the ibv/libc errno text attached,
// instead of being swallowed or turned into a bare bool. That satisfies the
// "errors must be checked and surfaced with useful context" requirement:
// every throw site names the call that failed, not just "something broke".
class VerbsError : public std::runtime_error {
 public:
  VerbsError(const std::string& what, int err)
      : std::runtime_error(what + ": " + std::strerror(err)), err_(err) {}
  int errno_value() const { return err_; }

 private:
  int err_;
};

// Most ibv_* setup calls (ibv_open_device, ibv_create_qp, ...) return a
// null pointer on failure and set errno. Call sites read as:
//   qp_ = check_ptr(ibv_create_qp(pd, &attr), "ibv_create_qp");
template <typename T>
T* check_ptr(T* ptr, const std::string& call) {
  if (ptr == nullptr) {
    throw VerbsError(call, errno);
  }
  return ptr;
}

// A different subset of ibv_* calls (ibv_modify_qp, ibv_query_port, ...)
// return the errno value directly as an int instead of touching the global
// errno. Same story, different check.
inline void check_rc(int rc, const std::string& call) {
  if (rc != 0) {
    throw VerbsError(call, rc);
  }
}

}  // namespace micro_ccl
