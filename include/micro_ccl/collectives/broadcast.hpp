#pragma once

#include <cstddef>

#include "micro_ccl/collectives/reduce_ops.hpp"
#include "micro_ccl/communicator.hpp"

namespace micro_ccl {

// Root rank sends buf (count elements of dtype) to every other rank,
// overwriting their copy of buf. Implemented as a flat fan-out -- root
// sends directly to each of the other world_size - 1 ranks in turn -- not
// a binomial tree. That is an O(P) latency choice, not O(log P); the
// right call for this library's scope (small rank counts, correctness and
// clarity prioritized over shaving broadcast latency at scale) but it is
// the first thing to change if broadcast needed to scale to many ranks: a
// binomial tree lets already-received ranks start relaying immediately
// instead of all waiting on root alone.
void broadcast(Communicator& comm, MemoryRegion& mr, void* buf, size_t count,
                Dtype dtype, int root);

}  // namespace micro_ccl
