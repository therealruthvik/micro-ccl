#pragma once

#include <cstddef>

#include "micro_ccl/collectives/reduce_ops.hpp"
#include "micro_ccl/communicator.hpp"

namespace micro_ccl {

// Ring allgather. `buf` holds world_size * send_count elements laid out as
// world_size contiguous per-rank segments; on entry this rank's own
// segment ([rank * send_count, (rank+1) * send_count)) must already hold
// its contribution, and on return every rank's segment is visible in
// every buf. Implemented as world_size - 1 ring steps -- structurally the
// allgather phase of allreduce_ring with no reduction step, since there is
// nothing to combine here, only to circulate: every segment is already
// "finished" the moment its owning rank produced it.
void allgather(Communicator& comm, MemoryRegion& mr, void* buf,
                size_t send_count, Dtype dtype);

}  // namespace micro_ccl
