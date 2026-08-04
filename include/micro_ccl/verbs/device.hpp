#pragma once

#include <infiniband/verbs.h>

#include <cstdint>
#include <string>
#include <vector>

namespace micro_ccl {

// Wraps an RDMA device (ibv_context) plus the one active port on it that
// we actually use. micro-ccl only ever talks to a single port on a single
// device per process, so folding "device + port" into one object keeps the
// rest of the codebase from having to pass (context, port_num) pairs
// everywhere.
//
// This class is also where the one piece of RoCE-vs-InfiniBand branching in
// the whole library lives: InfiniBand addresses peers by LID, RoCE
// addresses them by GID (RoCE encapsulates verbs traffic in UDP/IP, so it
// needs an IP-like address). Everything above this class -- QueuePair,
// transport, collectives -- only ever sees a Device and asks it "how do I
// address myself", so the RoCE/IB split does not leak upward. That is what
// lets the same collectives code run unmodified on Soft-RoCE, real RoCE, or
// real InfiniBand.
class Device {
 public:
  // dev_name empty => pick the first device with an active port.
  explicit Device(const std::string& dev_name = "");
  ~Device();

  // Move-only: this object owns an ibv_context*, a kernel resource handle.
  // Copying it would give two objects the same handle, and both destructors
  // would call ibv_close_device() on it -- a double-free. Deleting the copy
  // constructor/assignment makes that a compile error instead of a runtime
  // one. Move transfers ownership and nulls out the source, so only one
  // object ever closes the handle.
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&& other) noexcept;
  Device& operator=(Device&& other) noexcept;

  ibv_context* context() const { return context_; }
  uint8_t port_num() const { return port_num_; }
  const ibv_port_attr& port_attr() const { return port_attr_; }

  bool is_infiniband() const {
    return port_attr_.link_layer == IBV_LINK_LAYER_INFINIBAND;
  }

  // Valid only when !is_infiniband(). RoCE v2 is what Soft-RoCE and modern
  // NICs use (RoCE v1 is a deprecated Ethernet-only variant); we scan the
  // kernel's gid_attrs table to find the first RoCE v2 entry rather than
  // hardcoding an index, because the index varies by NIC/driver/how many
  // other GIDs (e.g. IPv6 link-local) happen to be registered first.
  int gid_index() const { return gid_index_; }
  const ibv_gid& gid() const { return gid_; }

  // Valid only when is_infiniband().
  uint16_t lid() const { return port_attr_.lid; }

  static std::vector<std::string> list_device_names();

 private:
  void find_active_port();
  void find_roce_v2_gid_index();

  ibv_context* context_ = nullptr;
  uint8_t port_num_ = 0;
  ibv_port_attr port_attr_{};
  int gid_index_ = -1;
  ibv_gid gid_{};
};

}  // namespace micro_ccl
