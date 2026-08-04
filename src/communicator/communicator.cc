#include "micro_ccl/communicator.hpp"

#include <stdexcept>

namespace micro_ccl {

Communicator::Communicator(Device& device, int rank, int world_size,
                            const std::string& root_host, uint16_t root_port)
    : device_(device),
      rank_(rank),
      world_size_(world_size),
      pd_(device_),
      send_cq_(device_),
      recv_cq_(device_),
      // Constructing bootstrap_ here blocks until every rank has joined
      // the rendezvous (rank 0 has accepted world_size - 1 connections, or
      // this rank has connected to rank 0) -- so by the time the
      // constructor body below runs, every rank is guaranteed reachable.
      bootstrap_(rank_, world_size_, root_host, root_port) {
  if (world_size_ < 1 || rank_ < 0 || rank_ >= world_size_) {
    throw std::invalid_argument("Communicator: invalid rank/world_size");
  }

  // One QP per peer, walked to INIT immediately -- INIT is local-only
  // state (port + access flags), so it does not need anything from the
  // peer yet and can happen before any bootstrap exchange.
  std::unordered_map<int, uint32_t> local_psn;
  std::unordered_map<int, QpEndpointInfo> outgoing;
  for (int peer = 0; peer < world_size_; ++peer) {
    if (peer == rank_) continue;
    auto [it, inserted] =
        qps_.emplace(peer, QueuePair(pd_, send_cq_, recv_cq_));
    QueuePair& qp = it->second;
    qp.modify_to_init(device_);
    uint32_t psn = QueuePair::generate_psn();
    local_psn[peer] = psn;
    outgoing[peer] = qp.local_info(device_, static_cast<uint32_t>(rank_), psn);
  }

  // Single round-trip: every rank's per-peer QP info goes to rank 0 and
  // comes back reflected to everyone. After this, each rank has exactly
  // what it needs to finish every one of its QPs independently.
  std::unordered_map<int, QpEndpointInfo> incoming =
      bootstrap_.exchange(outgoing);

  for (int peer = 0; peer < world_size_; ++peer) {
    if (peer == rank_) continue;
    QueuePair& qp = qps_.at(peer);
    qp.modify_to_rtr(device_, incoming.at(peer));
    qp.modify_to_rts(local_psn.at(peer));
  }
}

void Communicator::barrier() { bootstrap_.exchange({}); }

}  // namespace micro_ccl
