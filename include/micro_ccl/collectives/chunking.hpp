#pragma once

#include <cstddef>
#include <vector>

namespace micro_ccl {

// One rank's slice of a buffer that's being split num_chunks ways, in
// elements (not bytes) -- callers multiply by dtype_size() themselves,
// since chunking is a pure counting problem independent of element type.
struct ChunkRange {
  size_t offset;  // element index where this chunk starts
  size_t count;   // number of elements in this chunk, may be 0
};

// Splits total_count elements into num_chunks contiguous ranges as evenly
// as possible: when total_count doesn't divide evenly, the first
// (total_count % num_chunks) chunks get one extra element rather than
// leaving a leftover remainder chunk at the end. This keeps every rank's
// slice within 1 element of every other rank's, which matters for ring
// allreduce specifically -- an uneven split that dumped all the remainder
// into the last chunk would make one rank do disproportionately more
// send/recv/reduce work every single round.
//
// Always returns exactly num_chunks entries, summing to total_count; a
// chunk's count is 0 when num_chunks > total_count (fewer elements than
// ranks) -- callers must treat a 0-count chunk as "nothing to send/recv
// this step", not as an error.
std::vector<ChunkRange> compute_chunks(size_t total_count, size_t num_chunks);

}  // namespace micro_ccl
