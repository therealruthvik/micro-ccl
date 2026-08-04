// Benchmark harness for micro-ccl's own allreduce implementations. Sweeps
// message size from 8 bytes to 256 MiB in powers of two, running ring
// and/or recursive-doubling allreduce at each size, and emits one CSV row
// per (algorithm, size) pair.
//
// CSV schema (shared with bench_mpi_allreduce.cc's output so both can be
// concatenated and plotted together):
//   algo,dtype,op,world_size,size_bytes,count,iters,min_us,median_us,p99_us,bandwidth_gbps
//
// Only rank 0 writes CSV rows -- every rank participates in every
// collective call (they have to, it's a collective), but only rank 0's
// wall-clock timings are recorded and reported. This is the same
// methodology OSU's micro-benchmarks use: the collective's steps are
// synchronous round trips with every peer, so one rank's measured time is
// a faithful proxy for how long the whole operation took everywhere.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "micro_ccl/collectives.hpp"
#include "micro_ccl/communicator.hpp"
#include "micro_ccl/verbs.hpp"

using namespace micro_ccl;

namespace {

struct Args {
  int rank = -1;
  int world_size = 2;
  std::string root_host;
  uint16_t root_port = 20400;
  std::string algo = "both";      // ring | recdouble | both
  std::string dtype_str = "float32";
  std::string op_str = "sum";
  size_t min_size = 8;
  size_t max_size = 256ull * 1024 * 1024;
  std::string csv_out;
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
      return argv[++i];
    };
    if (flag == "--rank") a.rank = std::stoi(next());
    else if (flag == "--world-size") a.world_size = std::stoi(next());
    else if (flag == "--root-host") a.root_host = next();
    else if (flag == "--root-port") a.root_port = static_cast<uint16_t>(std::stoi(next()));
    else if (flag == "--algo") a.algo = next();
    else if (flag == "--dtype") a.dtype_str = next();
    else if (flag == "--op") a.op_str = next();
    else if (flag == "--min-size") a.min_size = std::stoull(next());
    else if (flag == "--max-size") a.max_size = std::stoull(next());
    else if (flag == "--csv-out") a.csv_out = next();
    else throw std::runtime_error("unknown flag: " + flag);
  }
  if (a.rank < 0 || a.root_host.empty()) {
    throw std::runtime_error(
        "usage: bench_allreduce --rank R --world-size N --root-host H "
        "[--root-port P] [--algo ring|recdouble|both] "
        "[--dtype float32|int32] [--op sum|min|max] "
        "[--min-size BYTES] [--max-size BYTES] [--csv-out FILE]");
  }
  return a;
}

Dtype parse_dtype(const std::string& s) {
  if (s == "float32") return Dtype::Float32;
  if (s == "int32") return Dtype::Int32;
  throw std::runtime_error("unknown dtype: " + s);
}

ReduceOp parse_op(const std::string& s) {
  if (s == "sum") return ReduceOp::Sum;
  if (s == "min") return ReduceOp::Min;
  if (s == "max") return ReduceOp::Max;
  throw std::runtime_error("unknown op: " + s);
}

// A fixed iteration count would make the 8-byte end of the sweep
// statistically noisy (too fast, dominated by clock resolution) and the
// 256 MiB end take forever (each iteration alone may be tens of
// milliseconds). Scaling iteration count down as message size grows keeps
// total sweep wall-clock time roughly bounded while still giving small
// sizes enough samples for a meaningful p99.
int choose_iters(size_t size_bytes) {
  if (size_bytes <= 64ull * 1024) return 200;
  if (size_bytes <= 1024ull * 1024) return 50;
  if (size_bytes <= 16ull * 1024 * 1024) return 20;
  return 5;
}
constexpr int kWarmup = 5;

struct Stats {
  double min_us, median_us, p99_us;
};

Stats compute_stats(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  size_t n = samples.size();
  Stats s;
  s.min_us = samples.front();
  s.median_us = samples[n / 2];
  s.p99_us = samples[static_cast<size_t>(0.99 * (n - 1))];
  return s;
}

struct Row {
  std::string algo;
  std::string dtype;
  std::string op;
  int world_size;
  size_t size_bytes;
  size_t count;
  int iters;
  Stats stats;
};

double bandwidth_gbps(size_t size_bytes, double median_us) {
  double seconds = median_us / 1e6;
  return (static_cast<double>(size_bytes) / seconds) / 1e9;
}

void write_row(std::ofstream& out, const Row& r) {
  out << r.algo << ',' << r.dtype << ',' << r.op << ',' << r.world_size
      << ',' << r.size_bytes << ',' << r.count << ',' << r.iters << ','
      << r.stats.min_us << ',' << r.stats.median_us << ',' << r.stats.p99_us
      << ',' << bandwidth_gbps(r.size_bytes, r.stats.median_us) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  try {
    args = parse_args(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }

  try {
    Dtype dtype = parse_dtype(args.dtype_str);
    ReduceOp op = parse_op(args.op_str);
    size_t elem_size = dtype_size(dtype);

    Device device;
    Communicator comm(device, args.rank, args.world_size, args.root_host,
                       args.root_port);

    // Pre-register once, for the largest size in the sweep, and reuse for
    // every (algorithm, size) combination below -- registration cost does
    // not belong inside a latency measurement.
    size_t max_count = args.max_size / elem_size;
    std::vector<uint8_t> data(args.max_size);
    std::vector<uint8_t> scratch(args.max_size);  // sized for recursive
                                                    // doubling's full-buffer
                                                    // requirement; ring uses
                                                    // only a prefix of it.
    MemoryRegion data_mr(comm.pd(), data.data(), data.size());
    MemoryRegion scratch_mr(comm.pd(), scratch.data(), scratch.size());

    std::ofstream csv;
    if (args.rank == 0 && !args.csv_out.empty()) {
      csv.open(args.csv_out);
      csv << "algo,dtype,op,world_size,size_bytes,count,iters,min_us,"
             "median_us,p99_us,bandwidth_gbps\n";
    }

    std::vector<std::string> algos;
    if (args.algo == "both") {
      algos = {"ring", "recdouble"};
    } else {
      algos = {args.algo};
    }

    for (size_t size_bytes = args.min_size; size_bytes <= args.max_size;
         size_bytes *= 2) {
      size_t count = size_bytes / elem_size;
      if (count == 0) continue;  // size smaller than one element, skip

      for (const auto& algo : algos) {
        if (algo == "recdouble" &&
            (args.world_size & (args.world_size - 1)) != 0) {
          // Skip silently rather than throwing -- a sweep run with a
          // non-power-of-two world size should still produce ring's rows.
          continue;
        }

        int iters = choose_iters(size_bytes);
        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(iters));

        int total_rounds = kWarmup + iters;
        comm.barrier();
        for (int it = 0; it < total_rounds; ++it) {
          // Reset to a known pattern every round so a correctness bug
          // (e.g. an algorithm that stops reducing after round 1) would
          // show up as wrong output if this harness checked values -- it
          // doesn't (that's the integration tests' job), but keeping the
          // pattern fresh avoids accidentally benchmarking a degenerate
          // all-zero or already-reduced buffer.
          std::fill(data.begin(), data.begin() + count * elem_size,
                     static_cast<uint8_t>(args.rank + it));

          auto t0 = std::chrono::steady_clock::now();
          if (algo == "ring") {
            allreduce_ring(comm, data_mr, data.data(), count, dtype, op,
                            scratch_mr, scratch.data(), max_count);
          } else {
            allreduce_recursive_doubling(comm, data_mr, data.data(), count,
                                          dtype, op, scratch_mr,
                                          scratch.data(), max_count);
          }
          auto t1 = std::chrono::steady_clock::now();

          if (it >= kWarmup) {
            samples.push_back(
                std::chrono::duration<double, std::micro>(t1 - t0).count());
          }
        }

        if (args.rank == 0) {
          Stats s = compute_stats(samples);
          std::printf(
              "algo=%-10s size=%10zu count=%9zu iters=%4d min=%9.2fus "
              "median=%9.2fus p99=%9.2fus bw=%7.3fGB/s\n",
              algo.c_str(), size_bytes, count, iters, s.min_us, s.median_us,
              s.p99_us, bandwidth_gbps(size_bytes, s.median_us));
          if (csv.is_open()) {
            write_row(csv, {algo, args.dtype_str, args.op_str,
                             args.world_size, size_bytes, count, iters, s});
          }
        }
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[rank %d] error: %s\n", args.rank, e.what());
    return 1;
  }
  return 0;
}
