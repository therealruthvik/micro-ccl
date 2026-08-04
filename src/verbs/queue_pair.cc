#include "micro_ccl/verbs/queue_pair.hpp"

#include <algorithm>
#include <cstring>
#include <random>

#include "micro_ccl/detail/error.hpp"

namespace micro_ccl {

QueuePair::QueuePair(ProtectionDomain& pd, CompletionQueue& send_cq,
                      CompletionQueue& recv_cq, uint32_t max_send_wr,
                      uint32_t max_recv_wr, uint32_t max_sge) {
  ibv_qp_init_attr attr{};
  attr.qp_type = IBV_QPT_RC;
  attr.send_cq = send_cq.native();
  attr.recv_cq = recv_cq.native();
  attr.cap.max_send_wr = max_send_wr;
  attr.cap.max_recv_wr = max_recv_wr;
  attr.cap.max_send_sge = max_sge;
  attr.cap.max_recv_sge = max_sge;
  // sq_sig_all left at 0 (default): we mark IBV_SEND_SIGNALED explicitly per
  // work request in the transport layer rather than forcing every send to
  // generate a completion. Signaling every WR is fine at our message rates,
  // but doing it explicitly documents the choice instead of relying on a
  // struct-zeroing side effect.

  qp_ = check_ptr(ibv_create_qp(pd.native(), &attr), "ibv_create_qp");
}

QueuePair::~QueuePair() {
  if (qp_ != nullptr) {
    ibv_destroy_qp(qp_);
  }
}

QueuePair::QueuePair(QueuePair&& other) noexcept : qp_(other.qp_) {
  other.qp_ = nullptr;
}

QueuePair& QueuePair::operator=(QueuePair&& other) noexcept {
  if (this != &other) {
    if (qp_ != nullptr) {
      ibv_destroy_qp(qp_);
    }
    qp_ = other.qp_;
    other.qp_ = nullptr;
  }
  return *this;
}

uint32_t QueuePair::generate_psn() {
  // thread_local: std::mt19937 carries ~2.5KB of state and seeding it is
  // not free, so we do not want to build a fresh one per call. thread_local
  // (rather than a plain static) avoids a data race if collectives ever set
  // up multiple QPs from different threads concurrently.
  static thread_local std::mt19937 rng(std::random_device{}());
  static thread_local std::uniform_int_distribution<uint32_t> dist(
      0, 0xFFFFFF);  // PSN is a 24-bit field
  return dist(rng);
}

void QueuePair::modify_to_init(const Device& device) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = device.port_num();
  // Allow the peer to target this QP with RDMA read/write once connected.
  // Two-sided send/recv (what point-to-point and the collectives use)
  // doesn't need these, but granting them costs nothing at setup time and
  // keeps the one-sided path available at the transport layer.
  attr.qp_access_flags =
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;

  int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
             IBV_QP_ACCESS_FLAGS;
  check_rc(ibv_modify_qp(qp_, &attr, mask), "ibv_modify_qp(INIT)");
}

void QueuePair::modify_to_rtr(const Device& local_device,
                               const QpEndpointInfo& remote) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTR;

  // Path MTU must not exceed either side's active port MTU. Each side
  // advertises its own active_mtu in QpEndpointInfo (set in local_info());
  // taking the min here means a mismatched pair of NICs/ports still
  // connects correctly instead of one side silently sending oversized
  // packets the other can't reassemble.
  auto local_mtu = static_cast<int>(local_device.port_attr().active_mtu);
  auto remote_mtu = static_cast<int>(remote.mtu);
  attr.path_mtu = static_cast<ibv_mtu>(std::min(local_mtu, remote_mtu));

  attr.dest_qp_num = remote.qp_num;
  attr.rq_psn = remote.psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;

  ibv_ah_attr& ah = attr.ah_attr;
  ah.port_num = local_device.port_num();
  ah.sl = 0;
  ah.src_path_bits = 0;

  if (local_device.is_infiniband()) {
    ah.is_global = 0;
    ah.dlid = remote.lid;
  } else {
    // RoCE has no LID concept; addressing works over the GRH (Global
    // Routing Header) instead, the same mechanism InfiniBand uses to route
    // between subnets. hop_limit=1 is fine here since Soft-RoCE and our
    // target deployments are single-hop L2/L3-adjacent.
    ah.is_global = 1;
    ah.grh.dgid = remote.gid;
    ah.grh.sgid_index = local_device.gid_index();
    ah.grh.hop_limit = 1;
    ah.grh.traffic_class = 0;
    ah.grh.flow_label = 0;
  }

  int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
             IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
             IBV_QP_MIN_RNR_TIMER;
  check_rc(ibv_modify_qp(qp_, &attr, mask), "ibv_modify_qp(RTR)");
}

void QueuePair::modify_to_rts(uint32_t local_psn) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = local_psn;
  attr.timeout = 14;     // ~2^14 * 4.096us ACK timeout before retry
  attr.retry_cnt = 7;    // max retransmit attempts on timeout
  attr.rnr_retry = 7;    // 7 = retry forever on receiver-not-ready
  attr.max_rd_atomic = 1;

  int mask = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
             IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
  check_rc(ibv_modify_qp(qp_, &attr, mask), "ibv_modify_qp(RTS)");
}

QpEndpointInfo QueuePair::local_info(const Device& device, uint32_t rank,
                                      uint32_t local_psn) const {
  QpEndpointInfo info{};
  info.rank = rank;
  info.qp_num = qp_->qp_num;
  info.psn = local_psn;
  info.port_num = device.port_num();
  info.mtu = static_cast<uint8_t>(device.port_attr().active_mtu);
  if (device.is_infiniband()) {
    info.lid = device.lid();
  } else {
    info.gid_index = static_cast<uint8_t>(device.gid_index());
    info.gid = device.gid();
  }
  return info;
}

}  // namespace micro_ccl
