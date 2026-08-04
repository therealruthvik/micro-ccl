// Unit tests for the chunk-splitting math ring allreduce and allgather
// both depend on. This is the kind of code where an off-by-one silently
// drops or double-counts an element instead of crashing -- worth pinning
// down with exact-value assertions, not just "it doesn't throw".
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <stdexcept>

#include "micro_ccl/collectives/chunking.hpp"

using namespace micro_ccl;

TEST_CASE("compute_chunks splits evenly when it divides exactly") {
  auto chunks = compute_chunks(12, 4);
  REQUIRE(chunks.size() == 4);
  for (const auto& c : chunks) {
    REQUIRE(c.count == 3);
  }
  REQUIRE(chunks[0].offset == 0);
  REQUIRE(chunks[1].offset == 3);
  REQUIRE(chunks[2].offset == 6);
  REQUIRE(chunks[3].offset == 9);
}

TEST_CASE("compute_chunks gives the remainder to the first chunks, one each") {
  // 10 elements over 3 chunks: base=3, remainder=1 -> sizes {4, 3, 3}.
  auto chunks = compute_chunks(10, 3);
  REQUIRE(chunks.size() == 3);
  REQUIRE(chunks[0].count == 4);
  REQUIRE(chunks[1].count == 3);
  REQUIRE(chunks[2].count == 3);
  REQUIRE(chunks[0].offset == 0);
  REQUIRE(chunks[1].offset == 4);
  REQUIRE(chunks[2].offset == 7);
}

TEST_CASE("compute_chunks chunk sizes never differ by more than one element") {
  // A ring/allgather step's cost is set by its largest chunk; an uneven
  // split would make one rank do disproportionately more work every round.
  auto chunks = compute_chunks(97, 8);  // deliberately not evenly divisible
  size_t total = 0;
  size_t min_count = chunks[0].count;
  size_t max_count = chunks[0].count;
  for (const auto& c : chunks) {
    total += c.count;
    min_count = std::min(min_count, c.count);
    max_count = std::max(max_count, c.count);
  }
  REQUIRE(total == 97);
  REQUIRE(max_count - min_count <= 1);
}

TEST_CASE("compute_chunks with more chunks than elements yields zero-size tail chunks") {
  auto chunks = compute_chunks(3, 8);
  REQUIRE(chunks.size() == 8);
  size_t total = 0;
  int nonzero = 0;
  for (const auto& c : chunks) {
    total += c.count;
    if (c.count > 0) ++nonzero;
  }
  REQUIRE(total == 3);
  REQUIRE(nonzero == 3);
  // The nonzero chunks must be the first three -- callers skip send/recv
  // on count==0 chunks, so which chunks are empty has to be predictable.
  REQUIRE(chunks[0].count == 1);
  REQUIRE(chunks[1].count == 1);
  REQUIRE(chunks[2].count == 1);
  REQUIRE(chunks[3].count == 0);
  REQUIRE(chunks[7].count == 0);
}

TEST_CASE("compute_chunks with a single chunk returns the whole range") {
  auto chunks = compute_chunks(100, 1);
  REQUIRE(chunks.size() == 1);
  REQUIRE(chunks[0].offset == 0);
  REQUIRE(chunks[0].count == 100);
}

TEST_CASE("compute_chunks with zero elements returns all-empty chunks") {
  auto chunks = compute_chunks(0, 4);
  REQUIRE(chunks.size() == 4);
  for (const auto& c : chunks) {
    REQUIRE(c.count == 0);
  }
}

TEST_CASE("compute_chunks rejects zero chunks") {
  REQUIRE_THROWS_AS(compute_chunks(10, 0), std::invalid_argument);
}
