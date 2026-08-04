#pragma once

#include <cstddef>

#include "micro_ccl/collectives/reduce_ops.hpp"
#include "micro_ccl/communicator.hpp"

namespace micro_ccl {

// Ring allreduce: the classic bandwidth-optimal collective. With P ranks,
// the buffer is split into P chunks; a "reduce-scatter" phase of P-1 steps
// passes each chunk once around the ring accumulating partial sums, after
// which every rank holds the complete reduction of exactly one chunk; a
// mirror-image "allgather" phase of P-1 steps then passes those completed
// chunks the rest of the way around so every rank ends up with all P
// chunks, fully reduced.
//
// Why this shape rather than something simpler (e.g. everyone sends to
// rank 0, rank 0 reduces, rank 0 broadcasts back)? Every rank in the ring
// only ever talks to its two ring neighbors, and every step moves exactly
// data_size/P bytes, regardless of P. Total data moved per rank across
// both phases is 2*(P-1)/P * data_size -- as P grows this approaches
// 2*data_size and never exceeds it, whereas the "everyone to rank 0"
// approach makes rank 0's link a bottleneck that gets worse as P grows.
// The cost of the ring's elegance is latency: 2*(P-1) sequential
// round-trip steps, each gated on the previous one finishing, which is
// why it loses to recursive-doubling (see allreduce_recursive_doubling.hpp)
// at small message sizes where those round trips' fixed latency dominates
// over actual data transfer time.
//
// `scratch_mr`/`scratch` is a second pre-registered buffer used as the
// landing zone for each incoming chunk during the reduce-scatter phase,
// before it's summed into `data`. It must be at least as large as the
// biggest single chunk (ceil(count / world_size) elements) -- passed in
// rather than allocated internally because registration is a setup-time
// cost this library does not want to pay per call (see MemoryRegion).
// Undersized scratch is a caller error and throws rather than silently
// truncating a chunk.
void allreduce_ring(Communicator& comm, MemoryRegion& data_mr, void* data,
                     size_t count, Dtype dtype, ReduceOp op,
                     MemoryRegion& scratch_mr, void* scratch,
                     size_t scratch_capacity_elems);

}  // namespace micro_ccl
