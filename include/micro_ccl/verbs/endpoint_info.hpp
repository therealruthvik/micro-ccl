#pragma once

#include <infiniband/verbs.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace micro_ccl {

// Everything one side needs to tell the other before an RC queue pair can
// move INIT -> RTR -> RTS: which QP to connect to, where to address it
// (LID for InfiniBand, GID for RoCE), the starting packet sequence number,
// the negotiated path MTU, and this rank's registered-buffer key/address
// (for the one-sided RDMA read/write path the transport layer exposes but
// collectives do not currently use). This struct is the entire payload of
// the out-of-band bootstrap exchange -- nothing else needs to cross the TCP
// side-channel.
struct QpEndpointInfo {
  uint32_t rank = 0;
  uint32_t qp_num = 0;
  uint32_t psn = 0;
  uint16_t lid = 0;       // meaningful when the device is InfiniBand
  uint8_t port_num = 0;
  uint8_t mtu = 0;        // ibv_mtu enum value, e.g. IBV_MTU_1024
  uint8_t gid_index = 0;  // meaningful when the device is RoCE
  ibv_gid gid{};          // meaningful when the device is RoCE
  uint32_t rkey = 0;
  uint64_t addr = 0;
  uint64_t buf_len = 0;
};

// Fixed-size, explicit-byte-order wire format for QpEndpointInfo. We do not
// just memcpy the struct onto the socket: struct layout (padding, field
// order) is a compiler/ABI detail, not a contract, and multi-byte integers
// need a defined byte order to be correct between two processes -- even
// though today both ends are x86_64 Linux, coding to "whatever memcpy
// happens to produce" is the kind of thing that bites you the first time
// someone builds one side differently. encode()/decode() give a stable,
// self-contained wire format independent of either.
constexpr size_t kEndpointInfoWireSize =
    4 + 4 + 4 + 2 + 1 + 1 + 1 + 16 + 4 + 8 + 8;  // = 53 bytes

void encode_endpoint_info(const QpEndpointInfo& info,
                           std::array<uint8_t, kEndpointInfoWireSize>& out);
QpEndpointInfo decode_endpoint_info(
    const std::array<uint8_t, kEndpointInfoWireSize>& in);

}  // namespace micro_ccl
