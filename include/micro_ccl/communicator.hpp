#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "micro_ccl/bootstrap.hpp"
#include "micro_ccl/verbs.hpp"

namespace micro_ccl {

// The multi-rank equivalent of what pingpong.cc did by hand for two
// processes: given a rank, a world size, and a bootstrap rendezvous
// address, stand up a full-mesh RC connection to every other rank (each
// rank ends up owning world_size - 1 QueuePairs, one per peer) and hand
// back a single object collectives can address peers through by rank
// number.
//
// Design choice worth naming: one shared send CompletionQueue and one
// shared receive CompletionQueue serve *all* of this rank's QPs, rather
// than a CQ per QP. A completion always carries the QP it came from
// (wc.qp_num), so nothing is lost by sharing, and it means collectives
// that are talking to several peers at once (ring allreduce talks to
// exactly 2: predecessor and successor) drain one queue instead of
// juggling several. This is the same pattern real MPI/NCCL-style libraries
// use, not a shortcut specific to this project.
//
// Full mesh (every rank connects to every other rank directly, O(N^2) QPs
// cluster-wide) is the right choice for the small scale this project
// targets. It is not what you'd want at real HPC scale -- see the ring
// algorithm's own design, which deliberately avoids needing a full mesh at
// the *algorithmic* level even though the underlying connections here
// happen to form one.
class Communicator {
 public:
  Communicator(Device& device, int rank, int world_size,
               const std::string& root_host, uint16_t root_port);

  int rank() const { return rank_; }
  int size() const { return world_size_; }

  ProtectionDomain& pd() { return pd_; }
  CompletionQueue& send_cq() { return send_cq_; }
  CompletionQueue& recv_cq() { return recv_cq_; }
  Device& device() { return device_; }

  // Throws std::out_of_range if peer_rank is this rank or not connected.
  QueuePair& qp(int peer_rank) { return qps_.at(peer_rank); }

  // Blocks until every rank has called barrier() for this round. Built on
  // the same bootstrap TCP channel used for connection setup (kept alive
  // for the Communicator's lifetime) -- reusing it here is simpler than
  // adding a second synchronization mechanism on the RDMA path, and
  // barriers are inherently a setup/coordination concern, not a data-path
  // one.
  void barrier();

 private:
  Device& device_;
  int rank_;
  int world_size_;
  ProtectionDomain pd_;
  CompletionQueue send_cq_;
  CompletionQueue recv_cq_;
  TcpBootstrap bootstrap_;
  std::unordered_map<int, QueuePair> qps_;
};

}  // namespace micro_ccl
