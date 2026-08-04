#include "micro_ccl/collectives/allreduce_recursive_doubling.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "micro_ccl/transport.hpp"

namespace micro_ccl {
namespace {
bool is_power_of_two(int p) { return p > 0 && (p & (p - 1)) == 0; }
}  // namespace

void allreduce_recursive_doubling(Communicator& comm, MemoryRegion& data_mr,
                                   void* data, size_t count, Dtype dtype,
                                   ReduceOp op, MemoryRegion& scratch_mr,
                                   void* scratch,
                                   size_t scratch_capacity_elems) {
  const int P = comm.size();
  const int R = comm.rank();
  const size_t elem_size = dtype_size(dtype);

  if (count * elem_size > data_mr.length()) {
    throw std::runtime_error(
        "allreduce_recursive_doubling: count exceeds registered data "
        "buffer length");
  }
  if (!is_power_of_two(P)) {
    throw std::runtime_error(
        "allreduce_recursive_doubling: world_size (" + std::to_string(P) +
        ") must be a power of two");
  }
  if (P == 1) {
    return;
  }
  if (scratch_capacity_elems < count) {
    throw std::runtime_error(
        "allreduce_recursive_doubling: scratch buffer (" +
        std::to_string(scratch_capacity_elems) +
        " elems) smaller than the full buffer (" + std::to_string(count) +
        " elems) -- recursive doubling exchanges the whole message every "
        "step, unlike ring's per-chunk scratch requirement");
  }

  const size_t bytes = count * elem_size;
  int steps = 0;
  for (int p = P; p > 1; p >>= 1) ++steps;

  for (int step = 0; step < steps; ++step) {
    int partner = R ^ (1 << step);

    // Full-buffer exchange with the partner at this step's distance, then
    // fold the two together. Both sides post identically (recv, send, wait
    // send, wait recv, reduce) -- there is no sender/receiver asymmetry in
    // recursive doubling the way there is in a tree broadcast, both ranks
    // are simultaneously sending and receiving with the same peer.
    transport::post_recv(comm.qp(partner), scratch_mr, scratch, bytes,
                          /*wr_id=*/0);
    transport::post_send(comm.qp(partner), data_mr, data, bytes,
                          /*wr_id=*/0);
    transport::wait_completion(comm.send_cq());
    ibv_wc wc = transport::wait_completion(comm.recv_cq());
    if (wc.byte_len != bytes) {
      throw std::runtime_error(
          "allreduce_recursive_doubling: recv size mismatch at step " +
          std::to_string(step) + ", expected " + std::to_string(bytes) +
          " got " + std::to_string(wc.byte_len) +
          " -- ranks disagree on message size");
    }
    reduce_inplace(data, scratch, count, dtype, op);
  }
}

}  // namespace micro_ccl
