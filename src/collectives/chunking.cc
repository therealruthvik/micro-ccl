#include "micro_ccl/collectives/chunking.hpp"

#include <stdexcept>

namespace micro_ccl {

std::vector<ChunkRange> compute_chunks(size_t total_count,
                                        size_t num_chunks) {
  if (num_chunks == 0) {
    throw std::invalid_argument("compute_chunks: num_chunks must be > 0");
  }

  size_t base = total_count / num_chunks;
  size_t remainder = total_count % num_chunks;

  std::vector<ChunkRange> chunks;
  chunks.reserve(num_chunks);
  size_t offset = 0;
  for (size_t i = 0; i < num_chunks; ++i) {
    size_t count = base + (i < remainder ? 1 : 0);
    chunks.push_back({offset, count});
    offset += count;
  }
  return chunks;
}

}  // namespace micro_ccl
