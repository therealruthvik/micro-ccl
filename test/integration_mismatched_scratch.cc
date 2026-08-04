// Integration test: confirms micro-ccl fails loudly on a caller sizing
// error instead of silently corrupting memory.
//
// The scenario: allreduce_ring's scratch buffer receives one chunk
// (roughly count/world_size elements) per reduce-scatter step before
// reduce_inplace() folds it into `data`. If a caller under-sizes that
// scratch buffer, letting the receive proceed would write past the end of
// it -- a real buffer overrun. allreduce_ring instead checks the size
// up front (see its "Undersized scratch is a caller error" comment) and
// throws before issuing a single send/recv, deterministically and without
// depending on any network timing.
//
// That determinism is exactly why this test intentionally does NOT test
// the "two ranks disagree about message size" scenario instead: that
// failure mode is real (and allreduce_ring does check wc.byte_len against
// the expected size on every receive -- see its recv-size-mismatch
// checks), but which side observes the error first, and how the RC
// connection's retry/teardown behaves, depends on hardware timing that
// isn't reproducible in an automated test. The scratch-size check below
// exercises the same "fail loudly, don't corrupt memory" contract with a
// fully deterministic, pre-network assertion instead.
//
// Needs a real RDMA device, same as integration_allreduce_4rank.
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "micro_ccl/collectives.hpp"
#include "micro_ccl/communicator.hpp"
#include "micro_ccl/verbs.hpp"

namespace {

using namespace micro_ccl;

constexpr int kWorldSize = 2;
constexpr size_t kCount = 1000;         // -> chunks of 500 elements each
constexpr size_t kTooSmallScratch = 10;  // far below the required 500
constexpr uint16_t kBasePort = 20320;

int run_rank(int rank) {
  try {
    Device device;
    Communicator comm(device, rank, kWorldSize, "127.0.0.1", kBasePort);

    std::vector<int32_t> data(kCount, rank + 1);
    std::vector<int32_t> scratch(kTooSmallScratch, 0);
    MemoryRegion data_mr(comm.pd(), data.data(),
                          data.size() * sizeof(int32_t));
    MemoryRegion scratch_mr(comm.pd(), scratch.data(),
                             scratch.size() * sizeof(int32_t));

    allreduce_ring(comm, data_mr, data.data(), kCount, Dtype::Int32,
                    ReduceOp::Sum, scratch_mr, scratch.data(),
                    scratch.size());

    // Reaching here means the undersized scratch buffer was NOT caught --
    // that is the failure condition this test exists to catch.
    std::fprintf(stderr,
                  "[rank %d] FAIL: allreduce_ring accepted an undersized "
                  "scratch buffer instead of throwing\n",
                  rank);
    return 1;
  } catch (const std::exception& e) {
    std::printf("[rank %d] correctly failed loudly: %s\n", rank, e.what());
    return 0;
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
      // _exit() (unlike exit()) does not flush stdio buffers -- without
      // this, every printf/fprintf above is silently dropped whenever
      // stdout/stderr are fully-buffered (e.g. piped, as under ctest or
      // this project's own Docker compile-check, rather than an
      // interactive terminal), and the test would still pass or fail
      // correctly but leave no diagnostic output explaining why.
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
