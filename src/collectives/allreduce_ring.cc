#include "micro_ccl/collectives/allreduce_ring.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "micro_ccl/collectives/chunking.hpp"
#include "micro_ccl/transport.hpp"

namespace micro_ccl {

void allreduce_ring(Communicator& comm, MemoryRegion& data_mr, void* data,
                     size_t count, Dtype dtype, ReduceOp op,
                     MemoryRegion& scratch_mr, void* scratch,
                     size_t scratch_capacity_elems) {
  const int P = comm.size();
  const int R = comm.rank();
  const size_t elem_size = dtype_size(dtype);

  if (count * elem_size > data_mr.length()) {
    throw std::runtime_error(
        "allreduce_ring: count exceeds registered data buffer length");
  }
  if (P == 1) {
    return;  // nothing to reduce with, data is already the answer
  }

  std::vector<ChunkRange> chunks = compute_chunks(count, static_cast<size_t>(P));
  size_t max_chunk_elems = 0;
  for (const auto& c : chunks) max_chunk_elems = std::max(max_chunk_elems, c.count);
  if (scratch_capacity_elems < max_chunk_elems) {
    throw std::runtime_error(
        "allreduce_ring: scratch buffer (" +
        std::to_string(scratch_capacity_elems) +
        " elems) smaller than largest chunk (" +
        std::to_string(max_chunk_elems) + " elems)");
  }

  auto mod_p = [P](int x) { return ((x % P) + P) % P; };
  const int send_to = mod_p(R + 1);
  const int recv_from = mod_p(R - 1);

  auto* data_bytes = static_cast<uint8_t*>(data);
  auto* scratch_bytes = static_cast<uint8_t*>(scratch);

  // --- Phase 1: reduce-scatter. After P-1 steps, this rank holds the
  // fully-reduced chunk at index (R + 1) % P; every other rank holds the
  // full reduction of a different chunk.
  for (int step = 0; step < P - 1; ++step) {
    const ChunkRange& send_chunk = chunks[mod_p(R - step)];
    const ChunkRange& recv_chunk = chunks[mod_p(R - step - 1)];

    if (recv_chunk.count > 0) {
      transport::post_recv(comm.qp(recv_from), scratch_mr, scratch_bytes,
                            recv_chunk.count * elem_size, /*wr_id=*/0);
    }
    if (send_chunk.count > 0) {
      transport::post_send(comm.qp(send_to), data_mr,
                            data_bytes + send_chunk.offset * elem_size,
                            send_chunk.count * elem_size, /*wr_id=*/0);
    }
    if (send_chunk.count > 0) {
      transport::wait_completion(comm.send_cq());
    }
    if (recv_chunk.count > 0) {
      ibv_wc wc = transport::wait_completion(comm.recv_cq());
      size_t expected = recv_chunk.count * elem_size;
      if (wc.byte_len != expected) {
        throw std::runtime_error(
            "allreduce_ring: reduce-scatter recv size mismatch, expected " +
            std::to_string(expected) + " got " +
            std::to_string(wc.byte_len) +
            " -- ranks disagree on chunk size");
      }
      reduce_inplace(data_bytes + recv_chunk.offset * elem_size,
                      scratch_bytes, recv_chunk.count, dtype, op);
    }
  }

  // --- Phase 2: allgather. Fully-reduced chunks are forwarded around the
  // ring, written directly into their final position in `data` -- no
  // reduction, no extra copy, the incoming DMA lands exactly where the
  // result needs to live.
  for (int step = 0; step < P - 1; ++step) {
    const ChunkRange& send_chunk = chunks[mod_p(R - step + 1)];
    const ChunkRange& recv_chunk = chunks[mod_p(R - step)];

    if (recv_chunk.count > 0) {
      transport::post_recv(comm.qp(recv_from), data_mr,
                            data_bytes + recv_chunk.offset * elem_size,
                            recv_chunk.count * elem_size, /*wr_id=*/0);
    }
    if (send_chunk.count > 0) {
      transport::post_send(comm.qp(send_to), data_mr,
                            data_bytes + send_chunk.offset * elem_size,
                            send_chunk.count * elem_size, /*wr_id=*/0);
    }
    if (send_chunk.count > 0) {
      transport::wait_completion(comm.send_cq());
    }
    if (recv_chunk.count > 0) {
      ibv_wc wc = transport::wait_completion(comm.recv_cq());
      size_t expected = recv_chunk.count * elem_size;
      if (wc.byte_len != expected) {
        throw std::runtime_error(
            "allreduce_ring: allgather recv size mismatch, expected " +
            std::to_string(expected) + " got " +
            std::to_string(wc.byte_len) +
            " -- ranks disagree on chunk size");
      }
    }
  }
}

}  // namespace micro_ccl
