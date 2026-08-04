#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "micro_ccl/verbs/endpoint_info.hpp"

namespace micro_ccl {

// Out-of-band rendezvous used once, before any RDMA connection exists, to
// let every rank learn every peer's QpEndpointInfo (QP number, GID/LID,
// starting PSN, MR rkey). This has to happen over an ordinary TCP socket:
// an RC queue pair cannot reach the RTR state without already knowing the
// peer's QP number and address, and there is no way to learn that except
// by asking over some channel that already works -- plain TCP is the
// obvious, boring choice, and it is a one-shot setup cost, not something
// on the data path, so "boring" is exactly right here.
//
// Topology: rank 0 is a rendezvous point every other rank connects to. Each
// rank ships rank 0 the QpEndpointInfo it wants delivered to each peer;
// rank 0 collects everyone's contributions and reflects the complete table
// back out to every rank. This keeps the protocol to a single well-known
// (host, port) -- no peer needs to know any other peer's address ahead of
// time, only rank 0's. The tradeoff is a star topology (one process
// converging N-1 connections) that would not scale to large clusters, but
// this project is explicitly targeting small-scale correctness
// demonstrations, not production bootstrap at hardware-cluster scale.
class TcpBootstrap {
 public:
  TcpBootstrap(int rank, int world_size, const std::string& root_host,
               uint16_t root_port);
  ~TcpBootstrap();

  TcpBootstrap(const TcpBootstrap&) = delete;
  TcpBootstrap& operator=(const TcpBootstrap&) = delete;

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }

  // `outgoing` maps destination rank -> the QpEndpointInfo this rank wants
  // that destination to receive (e.g. "here is the QP I created to talk to
  // you"). Blocks until every rank has submitted its full outgoing set,
  // then returns everything addressed to *this* rank, keyed by source
  // rank. Called once per QP-setup round; the ring/recursive-doubling
  // collectives do not use this at all, only initial connection setup does.
  std::unordered_map<int, QpEndpointInfo> exchange(
      const std::unordered_map<int, QpEndpointInfo>& outgoing);

 private:
  int rank_;
  int world_size_;
  int conn_fd_ = -1;          // non-root: our one connection to rank 0
  int listen_fd_ = -1;        // root only: accept socket
  std::vector<int> peer_fds_;  // root only: peer_fds_[r] is rank r's socket
};

}  // namespace micro_ccl
