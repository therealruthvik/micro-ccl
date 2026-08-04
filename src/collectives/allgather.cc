#include "micro_ccl/collectives/allgather.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "micro_ccl/transport.hpp"

namespace micro_ccl {

void allgather(Communicator& comm, MemoryRegion& mr, void* buf,
                size_t send_count, Dtype dtype) {
  const int P = comm.size();
  const int R = comm.rank();
  const size_t elem_size = dtype_size(dtype);
  const size_t total_bytes =
      static_cast<size_t>(P) * send_count * elem_size;

  if (total_bytes > mr.length()) {
    throw std::runtime_error(
        "allgather: world_size * send_count exceeds registered buffer "
        "length");
  }
  if (P == 1) {
    return;  // this rank's own segment is already the whole answer
  }

  auto mod_p = [P](int x) { return ((x % P) + P) % P; };
  const int send_to = mod_p(R + 1);
  const int recv_from = mod_p(R - 1);
  auto* bytes = static_cast<uint8_t*>(buf);
  const size_t seg_bytes = send_count * elem_size;

  for (int step = 0; step < P - 1; ++step) {
    int send_idx = mod_p(R - step);
    int recv_idx = mod_p(R - step - 1);

    transport::post_recv(comm.qp(recv_from), mr, bytes + recv_idx * seg_bytes,
                          seg_bytes, /*wr_id=*/0);
    transport::post_send(comm.qp(send_to), mr, bytes + send_idx * seg_bytes,
                          seg_bytes, /*wr_id=*/0);
    transport::wait_completion(comm.send_cq());
    ibv_wc wc = transport::wait_completion(comm.recv_cq());
    if (wc.byte_len != seg_bytes) {
      throw std::runtime_error(
          "allgather: recv size mismatch, expected " +
          std::to_string(seg_bytes) + " got " + std::to_string(wc.byte_len) +
          " -- ranks disagree on send_count");
    }
  }
}

}  // namespace micro_ccl
