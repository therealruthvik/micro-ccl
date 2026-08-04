#include "micro_ccl/verbs/endpoint_info.hpp"

#include <cstring>

namespace micro_ccl {
namespace {

// Small helpers that write/read big-endian ("network byte order") integers
// into a raw byte cursor. This is the same convention htonl/ntohl use;
// spelling it out by hand keeps encode/decode readable as "field, size,
// field, size" without a wall of macro calls.
void put_u16(uint8_t*& p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v >> 8);
  p[1] = static_cast<uint8_t>(v);
  p += 2;
}
void put_u32(uint8_t*& p, uint32_t v) {
  p[0] = static_cast<uint8_t>(v >> 24);
  p[1] = static_cast<uint8_t>(v >> 16);
  p[2] = static_cast<uint8_t>(v >> 8);
  p[3] = static_cast<uint8_t>(v);
  p += 4;
}
void put_u64(uint8_t*& p, uint64_t v) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    *p++ = static_cast<uint8_t>(v >> shift);
  }
}
void put_u8(uint8_t*& p, uint8_t v) { *p++ = v; }

uint16_t get_u16(const uint8_t*& p) {
  uint16_t v = (static_cast<uint16_t>(p[0]) << 8) | p[1];
  p += 2;
  return v;
}
uint32_t get_u32(const uint8_t*& p) {
  uint32_t v = (static_cast<uint32_t>(p[0]) << 24) |
               (static_cast<uint32_t>(p[1]) << 16) |
               (static_cast<uint32_t>(p[2]) << 8) | p[3];
  p += 4;
  return v;
}
uint64_t get_u64(const uint8_t*& p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v = (v << 8) | p[i];
  }
  p += 8;
  return v;
}
uint8_t get_u8(const uint8_t*& p) { return *p++; }

}  // namespace

void encode_endpoint_info(const QpEndpointInfo& info,
                           std::array<uint8_t, kEndpointInfoWireSize>& out) {
  uint8_t* p = out.data();
  put_u32(p, info.rank);
  put_u32(p, info.qp_num);
  put_u32(p, info.psn);
  put_u16(p, info.lid);
  put_u8(p, info.port_num);
  put_u8(p, info.mtu);
  put_u8(p, info.gid_index);
  // raw_gid is a 16-byte opaque address, not a single integer -- copied
  // byte-for-byte with no shift/endian conversion, same as copying an IPv6
  // address.
  std::memcpy(p, info.gid.raw, 16);
  p += 16;
  put_u32(p, info.rkey);
  put_u64(p, info.addr);
  put_u64(p, info.buf_len);
}

QpEndpointInfo decode_endpoint_info(
    const std::array<uint8_t, kEndpointInfoWireSize>& in) {
  QpEndpointInfo info{};
  const uint8_t* p = in.data();
  info.rank = get_u32(p);
  info.qp_num = get_u32(p);
  info.psn = get_u32(p);
  info.lid = get_u16(p);
  info.port_num = get_u8(p);
  info.mtu = get_u8(p);
  info.gid_index = get_u8(p);
  std::memcpy(info.gid.raw, p, 16);
  p += 16;
  info.rkey = get_u32(p);
  info.addr = get_u64(p);
  info.buf_len = get_u64(p);
  return info;
}

}  // namespace micro_ccl
