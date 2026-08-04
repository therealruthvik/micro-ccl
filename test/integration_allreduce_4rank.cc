// Integration test: 4 ranks, all on one host, all three collectives
// (allreduce_ring, allreduce_recursive_doubling, allgather) verified
// against an independently computed serial result.
//
// This needs a real RDMA device (Soft-RoCE's rdma_rxe or real hardware) --
// unlike the unit tests in this directory, it cannot run in an environment
// with no ibverbs device, and will fail fast with a clear error if none is
// present. Run it with `ctest -R integration_allreduce_4rank` (or directly)
// on a Soft-RoCE VM per docs/setup.md.
//
// Four ranks on one host means four processes, not four threads: an
// ibv_context is tied to a process the way a file descriptor is, so this
// forks before any of them touch verbs, and each child opens its own
// Device independently. fork()-ing *after* opening a device would be the
// wrong pattern -- the parent's ibv_context is not something a child can
// safely inherit and use concurrently -- so fork happens first, in main(),
// and run_rank() is where each child's own RDMA setup begins.
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "micro_ccl/collectives.hpp"
#include "micro_ccl/communicator.hpp"
#include "micro_ccl/verbs.hpp"

namespace {

constexpr int kWorldSize = 4;
constexpr size_t kCount = 4001;  // deliberately not divisible by kWorldSize,
                                  // to exercise the uneven-chunk path too
constexpr uint16_t kBasePort = 20310;

using namespace micro_ccl;

// Fills a buffer the same deterministic way on every rank so the expected
// result can be computed independently (see expected_sum below) without
// any rank needing to see another rank's data ahead of time.
void fill(std::vector<int32_t>& buf, int rank) {
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<int32_t>((rank + 1) * 3 - static_cast<int>(i % 11));
  }
}

int32_t expected_sum_at(size_t i) {
  int64_t total = 0;
  for (int r = 0; r < kWorldSize; ++r) {
    total += (r + 1) * 3 - static_cast<int64_t>(i % 11);
  }
  return static_cast<int32_t>(total);
}

bool check(const std::vector<int32_t>& buf, int rank, const char* label) {
  for (size_t i = 0; i < buf.size(); ++i) {
    int32_t expected = expected_sum_at(i);
    if (buf[i] != expected) {
      std::fprintf(stderr,
                    "[rank %d] %s mismatch at index %zu: got %d, expected "
                    "%d (serial computation)\n",
                    rank, label, i, buf[i], expected);
      return false;
    }
  }
  return true;
}

int run_rank(int rank) {
  try {
    Device device;
    Communicator comm(device, rank, kWorldSize, "127.0.0.1", kBasePort);

    size_t max_chunk = (kCount + kWorldSize - 1) / kWorldSize;
    std::vector<int32_t> data(kCount);
    std::vector<int32_t> scratch(max_chunk);
    MemoryRegion data_mr(comm.pd(), data.data(),
                          data.size() * sizeof(int32_t));
    MemoryRegion scratch_mr(comm.pd(), scratch.data(),
                             scratch.size() * sizeof(int32_t));

    // --- ring allreduce ---
    fill(data, rank);
    comm.barrier();
    allreduce_ring(comm, data_mr, data.data(), kCount, Dtype::Int32,
                    ReduceOp::Sum, scratch_mr, scratch.data(), scratch.size());
    if (!check(data, rank, "allreduce_ring")) return 1;
    std::printf("[rank %d] allreduce_ring OK\n", rank);

    // --- recursive-doubling allreduce (kWorldSize=4 is a power of two) ---
    // needs a full-size scratch buffer (see allreduce_recursive_doubling's
    // header comment for why its scratch requirement differs from ring's).
    std::vector<int32_t> rd_scratch(kCount);
    MemoryRegion rd_scratch_mr(comm.pd(), rd_scratch.data(),
                                rd_scratch.size() * sizeof(int32_t));
    fill(data, rank);
    comm.barrier();
    allreduce_recursive_doubling(comm, data_mr, data.data(), kCount,
                                  Dtype::Int32, ReduceOp::Sum, rd_scratch_mr,
                                  rd_scratch.data(), rd_scratch.size());
    if (!check(data, rank, "allreduce_recursive_doubling")) return 1;
    std::printf("[rank %d] allreduce_recursive_doubling OK\n", rank);

    // --- allgather ---
    size_t send_count = 64;
    std::vector<int32_t> gather_buf(send_count * kWorldSize, 0);
    MemoryRegion gather_mr(comm.pd(), gather_buf.data(),
                            gather_buf.size() * sizeof(int32_t));
    for (size_t i = 0; i < send_count; ++i) {
      gather_buf[rank * send_count + i] =
          static_cast<int32_t>(rank * 1000 + i);
    }
    comm.barrier();
    allgather(comm, gather_mr, gather_buf.data(), send_count, Dtype::Int32);
    for (int r = 0; r < kWorldSize; ++r) {
      for (size_t i = 0; i < send_count; ++i) {
        int32_t expected = static_cast<int32_t>(r * 1000 + i);
        int32_t got = gather_buf[r * send_count + i];
        if (got != expected) {
          std::fprintf(stderr,
                        "[rank %d] allgather mismatch: segment %d index "
                        "%zu got %d expected %d\n",
                        rank, r, i, got, expected);
          return 1;
        }
      }
    }
    std::printf("[rank %d] allgather OK\n", rank);

    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[rank %d] error: %s\n", rank, e.what());
    return 1;
  }
}

}  // namespace

int main() {
  std::vector<pid_t> pids;
  for (int rank = 0; rank < kWorldSize; ++rank) {
    pid_t pid = fork();
    if (pid < 0) {
      std::perror("fork");
      return 1;
    }
    if (pid == 0) {
      int code = run_rank(rank);
      // _exit() skips flushing stdio buffers (unlike exit()) -- without
      // this, buffered printf/fprintf output is silently dropped whenever
      // stdout/stderr aren't line-buffered, e.g. under ctest or Docker.
      std::fflush(stdout);
      std::fflush(stderr);
      _exit(code);
    }
    pids.push_back(pid);
  }

  int overall = 0;
  for (pid_t pid : pids) {
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      overall = 1;
    }
  }
  return overall;
}
