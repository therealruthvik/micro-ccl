#include "micro_ccl/verbs/device.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "micro_ccl/detail/error.hpp"

namespace micro_ccl {

std::vector<std::string> Device::list_device_names() {
  int num_devices = 0;
  ibv_device** list = ibv_get_device_list(&num_devices);
  // ibv_get_device_list returns NULL both on error (errno set) and on the
  // legitimate "zero devices" case (num_devices == 0). Only the former is
  // an error we should raise.
  if (list == nullptr && num_devices < 0) {
    throw VerbsError("ibv_get_device_list", errno);
  }
  std::vector<std::string> names;
  for (int i = 0; i < num_devices; ++i) {
    names.emplace_back(ibv_get_device_name(list[i]));
  }
  ibv_free_device_list(list);
  return names;
}

Device::Device(const std::string& dev_name) {
  int num_devices = 0;
  ibv_device** list = check_ptr(ibv_get_device_list(&num_devices),
                                 "ibv_get_device_list");

  ibv_device* target = nullptr;
  if (dev_name.empty()) {
    if (num_devices == 0) {
      ibv_free_device_list(list);
      throw std::runtime_error(
          "no RDMA devices found (is rdma_rxe loaded? see docs/setup.md)");
    }
    target = list[0];
  } else {
    for (int i = 0; i < num_devices; ++i) {
      if (dev_name == ibv_get_device_name(list[i])) {
        target = list[i];
        break;
      }
    }
    if (target == nullptr) {
      ibv_free_device_list(list);
      throw std::runtime_error("RDMA device not found: " + dev_name);
    }
  }

  // ibv_open_device() dups the device handle internally, so it is safe to
  // free the list right after this call -- context_ does not depend on
  // `list` staying alive.
  context_ = check_ptr(ibv_open_device(target), "ibv_open_device");
  ibv_free_device_list(list);

  try {
    find_active_port();
    if (!is_infiniband()) {
      find_roce_v2_gid_index();
    }
  } catch (...) {
    ibv_close_device(context_);
    throw;
  }
}

Device::~Device() {
  if (context_ != nullptr) {
    ibv_close_device(context_);
  }
}

Device::Device(Device&& other) noexcept
    : context_(other.context_),
      port_num_(other.port_num_),
      port_attr_(other.port_attr_),
      gid_index_(other.gid_index_),
      gid_(other.gid_) {
  // Null out the source's handle so its destructor becomes a no-op. Without
  // this, both the moved-from and moved-to Device would call
  // ibv_close_device() on the same context at scope exit.
  other.context_ = nullptr;
}

Device& Device::operator=(Device&& other) noexcept {
  if (this != &other) {
    if (context_ != nullptr) {
      ibv_close_device(context_);
    }
    context_ = other.context_;
    port_num_ = other.port_num_;
    port_attr_ = other.port_attr_;
    gid_index_ = other.gid_index_;
    gid_ = other.gid_;
    other.context_ = nullptr;
  }
  return *this;
}

void Device::find_active_port() {
  ibv_device_attr dev_attr{};
  check_rc(ibv_query_device(context_, &dev_attr), "ibv_query_device");

  for (uint8_t port = 1; port <= dev_attr.phys_port_cnt; ++port) {
    ibv_port_attr attr{};
    check_rc(ibv_query_port(context_, port, &attr), "ibv_query_port");
    if (attr.state == IBV_PORT_ACTIVE) {
      port_num_ = port;
      port_attr_ = attr;
      return;
    }
  }
  throw std::runtime_error(
      "no active port on device (check `ibv_devinfo`, `rdma link show`)");
}

void Device::find_roce_v2_gid_index() {
  // rdma-core does not expose "give me the RoCE v2 GID index" as a single
  // verbs call -- ibv_query_gid() only returns the GID *value* at an index
  // you already picked, not its type. The type (RoCE v1 vs RoCE v2 vs IB)
  // lives in a sysfs side-table the kernel maintains alongside the verbs
  // device. We scan it here so the rest of the library never hardcodes a
  // GID index, which would silently break on NICs where index 0 is an IPv6
  // link-local entry instead of RoCE v2.
  const std::string dev_name = ibv_get_device_name(context_->device);
  const std::string base = "/sys/class/infiniband/" + dev_name +
                            "/ports/" + std::to_string(port_num_) +
                            "/gid_attrs/types/";

  for (int idx = 0; idx < 256; ++idx) {
    std::ifstream f(base + std::to_string(idx));
    if (!f.is_open()) {
      // Sparse index range -- gaps are normal, keep scanning.
      continue;
    }
    std::string type;
    std::getline(f, type);
    if (type.find("RoCE v2") != std::string::npos) {
      ibv_gid gid{};
      check_rc(ibv_query_gid(context_, port_num_, idx, &gid),
               "ibv_query_gid");
      gid_index_ = idx;
      gid_ = gid;
      return;
    }
  }
  throw std::runtime_error(
      "no RoCE v2 GID found on " + dev_name + " port " +
      std::to_string(port_num_) +
      " (Soft-RoCE should always expose one -- check `rdma link show`)");
}

}  // namespace micro_ccl
