#pragma once

#include <cstddef>

#include "micro_ccl/collectives/reduce_ops.hpp"
#include "micro_ccl/communicator.hpp"

namespace micro_ccl {

// Recursive-doubling allreduce. Requires world_size to be a power of two
// -- see the note below on why, and what generalizing costs.
//
// Where ring allreduce moves data/P bytes per step across 2*(P-1) steps,
// recursive doubling moves the *entire* buffer every step, but only needs
// log2(P) steps: at step k, each rank exchanges its full current buffer
// with the partner rank found by flipping bit k of its own rank (rank XOR
// 2^k), and reduces the two together. After log2(P) steps every rank has
// independently combined contributions from all P ranks -- there is no
// separate allgather phase, because exchanging full buffers rather than
// chunks means each rank already ends up with the complete answer, not
// just its slice of it.
//
// This is why the two algorithms trade places depending on message size:
// recursive doubling's cost is dominated by log2(P) round trips of fixed
// per-message overhead (small win at small sizes, since ring pays 2*(P-1)
// round trips for the same data), while ring's cost is dominated by actual
// bytes moved (small win at large sizes, since recursive doubling
// re-transmits the *whole* buffer at every one of its log2(P) steps
// instead of 1/P of it).
//
// Power-of-two requirement: the "rank XOR 2^k" partner-finding trick only
// produces a valid, deadlock-free pairing when every rank has a distinct
// bit pattern across exactly log2(P) bits, i.e. P is a power of two. Real
// MPI implementations generalize this with an "extra ranks" preprocessing
// step (odd ranks above the largest power-of-two <= P first fold their
// data into a partner, sit out the recursive-doubling core, then get the
// result relayed back) -- deliberately not implemented here to keep the
// algorithm's structure legible; see the README's limitations section.
void allreduce_recursive_doubling(Communicator& comm, MemoryRegion& data_mr,
                                   void* data, size_t count, Dtype dtype,
                                   ReduceOp op, MemoryRegion& scratch_mr,
                                   void* scratch,
                                   size_t scratch_capacity_elems);

}  // namespace micro_ccl
