// Stage 2 deliverable: the foundational correctness test for the whole
// project. Two processes (rank 0 and rank 1), one RC queue pair between
// them, one message echoed back and forth. Everything above this file --
// the RAII layer, the bootstrap protocol, the transport wrappers -- exists
// to make this loop possible and correct. If this is wrong, everything
// built on top of it (collectives) is wrong too, so it also doubles as a
// manual correctness check: it verifies the echoed bytes match what was
// sent, not just that "some completion happened".
//
// Run on two Ubuntu/Soft-RoCE VMs, e.g.:
//   VM-A (rank 0, also the bootstrap rendezvous point):
//     ./pingpong --rank 0 --world-size 2 --root-host 10.0.0.1 --root-port 20200
//   VM-B (rank 1):
//     ./pingpong --rank 1 --world-size 2 --root-host 10.0.0.1 --root-port 20200

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "micro_ccl/bootstrap.hpp"
#include "micro_ccl/transport.hpp"
#include "micro_ccl/verbs.hpp"

namespace {

struct Args {
  int rank = -1;
  int world_size = 2;
  std::string root_host;
  uint16_t root_port = 20200;
  int iters = 1000;
  int warmup = 100;
  size_t msg_size = 4096;
};

// Hand-rolled flag parsing: this is a two-node smoke test, not a CLI tool
// with a long-term interface, so pulling in a flags library would be
// dependency weight for no real benefit -- the project's own constraints
// call for "no exotic dependencies".
Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + flag);
      }
      return argv[++i];
    };
    if (flag == "--rank") {
      args.rank = std::stoi(next());
    } else if (flag == "--world-size") {
      args.world_size = std::stoi(next());
    } else if (flag == "--root-host") {
      args.root_host = next();
    } else if (flag == "--root-port") {
      args.root_port = static_cast<uint16_t>(std::stoi(next()));
    } else if (flag == "--iters") {
      args.iters = std::stoi(next());
    } else if (flag == "--warmup") {
      args.warmup = std::stoi(next());
    } else if (flag == "--size") {
      args.msg_size = static_cast<size_t>(std::stoul(next()));
    } else {
      throw std::runtime_error("unknown flag: " + flag);
    }
  }
  if (args.rank < 0 || args.root_host.empty()) {
    throw std::runtime_error(
        "usage: pingpong --rank R --world-size N --root-host H "
        "[--root-port P] [--iters N] [--warmup N] [--size BYTES]");
  }
  return args;
}

void fill_pattern(std::vector<uint8_t>& buf, int rank, int iter) {
  // Content depends on both rank and iteration so a bug that echoes stale
  // data from a previous iteration (or the wrong peer's buffer) shows up as
  // a mismatch instead of silently passing.
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<uint8_t>((rank * 131 + iter * 17 + i) & 0xFF);
  }
}

struct Stats {
  double min_us, median_us, p99_us;
};

Stats compute_stats(std::vector<double> samples_us) {
  std::sort(samples_us.begin(), samples_us.end());
  size_t n = samples_us.size();
  Stats s;
  s.min_us = samples_us.front();
  s.median_us = samples_us[n / 2];
  s.p99_us = samples_us[static_cast<size_t>(0.99 * (n - 1))];
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace micro_ccl;

  Args args;
  try {
    args = parse_args(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }

  try {
    // --- Stage-2-relevant setup: one Device, one PD, one pair of CQs, one
    // QP. This is deliberately minimal -- broadcast/allreduce/etc. do not
    // exist yet at this stage, only the single link two ranks need.
    Device device;
    ProtectionDomain pd(device);
    CompletionQueue send_cq(device);
    CompletionQueue recv_cq(device);
    QueuePair qp(pd, send_cq, recv_cq);

    // Pre-registered, reused for every iteration -- registration is a
    // one-time setup cost (see MemoryRegion's comment), and the data path
    // below never allocates or copies into a staging buffer.
    std::vector<uint8_t> send_buf(args.msg_size);
    std::vector<uint8_t> recv_buf(args.msg_size);
    MemoryRegion send_mr(pd, send_buf.data(), send_buf.size());
    MemoryRegion recv_mr(pd, recv_buf.data(), recv_buf.size());

    qp.modify_to_init(device);
    uint32_t local_psn = QueuePair::generate_psn();

    // --- Bootstrap: this is the only part of the program that touches
    // TCP, and it happens entirely before modify_to_rtr. Once the RDMA
    // side takes over, this socket is never touched again.
    TcpBootstrap bootstrap(args.rank, args.world_size, args.root_host,
                            args.root_port);
    int peer_rank = 1 - args.rank;
    auto local = qp.local_info(device, static_cast<uint32_t>(args.rank),
                                local_psn);
    auto incoming = bootstrap.exchange({{peer_rank, local}});
    const QpEndpointInfo& remote = incoming.at(peer_rank);

    qp.modify_to_rtr(device, remote);
    qp.modify_to_rts(local_psn);
    std::printf("[rank %d] QP %u connected to peer QP %u, RTS\n", args.rank,
                qp.qp_num(), remote.qp_num);

    std::vector<double> rtt_us;
    rtt_us.reserve(static_cast<size_t>(args.iters));

    int total_rounds = args.warmup + args.iters;
    for (int iter = 0; iter < total_rounds; ++iter) {
      bool measured = iter >= args.warmup;
      auto t0 = std::chrono::steady_clock::now();

      if (args.rank == 0) {
        fill_pattern(send_buf, args.rank, iter);
        transport::post_recv(qp, recv_mr, recv_buf.data(), recv_buf.size(),
                              /*wr_id=*/1);
        transport::send(qp, send_cq, send_mr, send_buf.data(),
                         send_buf.size());
        ibv_wc wc = transport::wait_completion(recv_cq);
        if (wc.byte_len != send_buf.size()) {
          throw std::runtime_error(
              "pingpong: echo size mismatch, sent " +
              std::to_string(send_buf.size()) + " got " +
              std::to_string(wc.byte_len));
        }
        if (std::memcmp(send_buf.data(), recv_buf.data(), send_buf.size()) !=
            0) {
          throw std::runtime_error(
              "pingpong: echoed data does not match what was sent "
              "(iteration " +
              std::to_string(iter) + ")");
        }
      } else {
        size_t got = transport::recv(qp, recv_cq, recv_mr, recv_buf.data(),
                                      recv_buf.size());
        if (got != args.msg_size) {
          throw std::runtime_error(
              "pingpong: rank 1 received unexpected size " +
              std::to_string(got) + ", expected " +
              std::to_string(args.msg_size));
        }
        // Echo the exact bytes back -- no local buffer manipulation, this
        // is the same registered recv_buf handed straight to post_send.
        transport::send(qp, send_cq, recv_mr, recv_buf.data(), got);
      }

      auto t1 = std::chrono::steady_clock::now();
      if (measured && args.rank == 0) {
        rtt_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    if (args.rank == 0) {
      Stats s = compute_stats(rtt_us);
      std::printf(
          "[rank 0] %zu-byte round trip over %d iters: min=%.2fus "
          "median=%.2fus p99=%.2fus\n",
          args.msg_size, args.iters, s.min_us, s.median_us, s.p99_us);
    }
    std::printf("[rank %d] pingpong OK\n", args.rank);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[rank %d] error: %s\n", args.rank, e.what());
    return 1;
  }
  return 0;
}
