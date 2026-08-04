// Comparison-mode benchmark: runs the identical size sweep against
// OpenMPI's MPI_Allreduce so its numbers sit directly next to
// bench_allreduce's ring/recursive-doubling results. Same CSV schema
// (algo column reads "openmpi" here), same size range, same warmup/iters
// policy, same "only rank 0's wall-clock time is recorded" methodology --
// methodology has to match bench_allreduce's or the two CSVs are not
// actually comparable, only superficially similar-looking.
//
// Build only happens if CMake found an MPI install (see bench/CMakeLists.txt);
// run with `mpirun -np N ./bench_mpi_allreduce ...`.
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
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
    if (flag == "--dtype") a.dtype_str = next();
    else if (flag == "--op") a.op_str = next();
    else if (flag == "--min-size") a.min_size = std::stoull(next());
    else if (flag == "--max-size") a.max_size = std::stoull(next());
    else if (flag == "--csv-out") a.csv_out = next();
    else throw std::runtime_error("unknown flag: " + flag);
  }
  return a;
}

MPI_Datatype parse_mpi_dtype(const std::string& s, size_t& elem_size) {
  if (s == "float32") { elem_size = sizeof(float); return MPI_FLOAT; }
  if (s == "int32") { elem_size = sizeof(int32_t); return MPI_INT32_T; }
  throw std::runtime_error("unknown dtype: " + s);
}

MPI_Op parse_mpi_op(const std::string& s) {
  if (s == "sum") return MPI_SUM;
  if (s == "min") return MPI_MIN;
  if (s == "max") return MPI_MAX;
  throw std::runtime_error("unknown op: " + s);
}

// Mirrors bench_allreduce.cc's choose_iters() exactly -- see that file for
// the rationale. Kept in sync by hand since these are two small standalone
// binaries with genuinely different dependencies (verbs+bootstrap vs MPI),
// not something worth introducing a shared header for.
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

double bandwidth_gbps(size_t size_bytes, double median_us) {
  double seconds = median_us / 1e6;
  return (static_cast<double>(size_bytes) / seconds) / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int exit_code = 0;
  try {
    Args args = parse_args(argc, argv);
    size_t elem_size = 0;
    MPI_Datatype mpi_dtype = parse_mpi_dtype(args.dtype_str, elem_size);
    MPI_Op mpi_op = parse_mpi_op(args.op_str);

    std::vector<uint8_t> send_buf(args.max_size);
    std::vector<uint8_t> recv_buf(args.max_size);

    std::ofstream csv;
    if (rank == 0 && !args.csv_out.empty()) {
      csv.open(args.csv_out);
      csv << "algo,dtype,op,world_size,size_bytes,count,iters,min_us,"
             "median_us,p99_us,bandwidth_gbps\n";
    }

    for (size_t size_bytes = args.min_size; size_bytes <= args.max_size;
         size_bytes *= 2) {
      size_t count = size_bytes / elem_size;
      if (count == 0) continue;

      int iters = choose_iters(size_bytes);
      std::vector<double> samples;
      samples.reserve(static_cast<size_t>(iters));

      MPI_Barrier(MPI_COMM_WORLD);
      for (int it = 0; it < kWarmup + iters; ++it) {
        std::memset(send_buf.data(), rank + it, count * elem_size);

        double t0 = MPI_Wtime();
        MPI_Allreduce(send_buf.data(), recv_buf.data(),
                       static_cast<int>(count), mpi_dtype, mpi_op,
                       MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (it >= kWarmup) {
          samples.push_back((t1 - t0) * 1e6);
        }
      }

      if (rank == 0) {
        Stats s = compute_stats(samples);
        std::printf(
            "algo=%-10s size=%10zu count=%9zu iters=%4d min=%9.2fus "
            "median=%9.2fus p99=%9.2fus bw=%7.3fGB/s\n",
            "openmpi", size_bytes, count, iters, s.min_us, s.median_us,
            s.p99_us, bandwidth_gbps(size_bytes, s.median_us));
        if (csv.is_open()) {
          csv << "openmpi," << args.dtype_str << ',' << args.op_str << ','
              << world_size << ',' << size_bytes << ',' << count << ','
              << iters << ',' << s.min_us << ',' << s.median_us << ','
              << s.p99_us << ',' << bandwidth_gbps(size_bytes, s.median_us)
              << '\n';
        }
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[rank %d] error: %s\n", rank, e.what());
    exit_code = 1;
  }

  MPI_Finalize();
  return exit_code;
}
