// SPDX-License-Identifier: Apache-2.0
// CPU SIMT shim; executes actual kernel helpers at 32 threads, not NPU timing.
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cassert>
#include <climits>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <thread>
#include <vector>
#define __gm__
#define __ubuf__
#define __simt_callee__
#define __aicore__
namespace sparse_kv_plan_config {
constexpr unsigned PLAN_THREADS = 32, PLAN_WARP_SIZE = 32, PLAN_WARP_COUNT = 1;
constexpr unsigned PLAN_SCAN_STORAGE_ELEMENTS = 16, PLAN_ROW_UB_LIMIT_ELEMENTS = 28672;
}
thread_local struct {
  unsigned x;
} threadIdx;
std::barrier team(32);
void asc_syncthreads() { team.arrive_and_wait(); }
int32_t asc_atomic_cas(int32_t* p, int32_t expected, int32_t desired) {
  std::atomic_ref<int32_t>(*p).compare_exchange_strong(expected, desired);
  return expected;
}
void asc_atomic_min(int32_t* p, int32_t v) {
  std::atomic_ref<int32_t> a(*p);
  auto old = a.load();
  while (old > v && !a.compare_exchange_weak(old, v)) {
  }
}
void asc_atomic_max(int32_t* p, int32_t v) {
  std::atomic_ref<int32_t> a(*p);
  auto old = a.load();
  while (old < v && !a.compare_exchange_weak(old, v)) {
  }
}
int shuffle_values[32];
int32_t asc_shfl(int32_t v, uint32_t src, int32_t width) {
  shuffle_values[threadIdx.x] = v;
  asc_syncthreads();
  int result = shuffle_values[src];
  asc_syncthreads();
  return result;
}
int32_t asc_shfl_up(int32_t v, uint32_t offset, int32_t width) {
  unsigned lane = threadIdx.x;
  return asc_shfl(v, lane >= offset ? lane - offset : lane, width);
}
// INSERT_KERNEL
}
int main() {
  std::mt19937 rng(1967);
  int checks = 0, shared = 0, overflow = 0;
  for (int C : {2, 4, 16, 64}) {
    const int T = std::max(1, C / 2), R = 8, N = 128, Q = 4, BS = 2;
    std::vector<int64_t> ids{100, 200, 300, 400}, last(R, -1);
    std::vector<int> top(R * T, -1), stable(R), visible(R), req(R), bt(Q * N / BS), st(R * C, -1), lru(R * C);
    std::vector<int> cur(R * T, -1), counts(R), mt(R * T), ms(R * T), modes(R), owners(R), pages(R * C / BS);
    std::vector<int> kv(R * C, -1), kvLayer2(R * C, -1), source(Q * N), scan(16), scratch;
    int H = 32;
    while (H < 2 * T) H *= 2;
    int J = 32;
    while (J < 2 * C) J *= 2;
    scratch.resize(std::max(3 * H + C + T, 2 * J + C));
    for (int i = 0; i < R * C; ++i) lru[i] = i % C;
    std::iota(bt.begin(), bt.end(), 0);
    for (int step = 0; step < 70; ++step) {
      if (step % 13 == 0) {
        ++ids[0];
        for (int t = 0; t < N; ++t) source[t] = step * 10000 + t;
      }
      const int width = step % 4 + 1, active = 1 + rng() % R;
      for (int row = 0; row < active; ++row) {
        req[row] = (row / width) % Q;
        stable[row] = 20;
        visible[row] = 21 + row % width;
        for (int t = 20; t < N; ++t) source[req[row] * N + t] = step * 10000 + req[row] * 100 + t;
        std::set<int> chosen;
        while ((int)chosen.size() < T) chosen.insert(rng() % N);
        int p = 0;
        for (int t : chosen) top[row * T + p++] = t;
        // alternate high-overlap and overflowing requests, including future/invalid entries
        if (step % 3 == 0)
          for (int j = 0; j < T; ++j) top[row * T + j] = j % (C / 2 + 1);
        if (step % 3 == 1)
          for (int j = 0; j < T; ++j) top[row * T + j] = (row % width) * T + j;
        if (step % 7 == 0) top[row * T] = -1;
        if (step % 9 == 0 && T > 1) top[row * T + 1] = top[row * T];
      }
      JointPlanArgs a{ids.data(),
                      last.data(),
                      top.data(),
                      stable.data(),
                      visible.data(),
                      req.data(),
                      bt.data(),
                      st.data(),
                      lru.data(),
                      cur.data(),
                      counts.data(),
                      mt.data(),
                      ms.data(),
                      modes.data(),
                      owners.data(),
                      pages.data(),
                      (unsigned)T,
                      (unsigned)C,
                      (unsigned)H,
                      (unsigned)J,
                      N,
                      BS,
                      Q,
                      N / BS,
                      Q * N / BS};
      for (int first = 0; first < active;) {
        int end = first + 1;
        while (end < active && req[end] == req[first]) ++end;
        std::vector<std::thread> threads;
        for (unsigned t = 0; t < 32; ++t)
          threads.emplace_back([&, t, first, end] {
            threadIdx.x = t;
            ProcessRequestRun(a, first, end, scratch.data(), scan.data());
          });
        for (auto& th : threads) th.join();
        std::set<int> need;
        for (int row = first; row < end; ++row)
          for (int j = 0; j < T; ++j)
            if (JointTokenValid(a, row, top[row * T + j])) need.insert(top[row * T + j]);
        const bool join = end - first > 1 && need.size() <= (size_t)C;
        if (join)
          ++shared;
        else if (end - first > 1)
          ++overflow;
        std::set<int> transferred;
        for (int row = first; row < end; ++row) {
          assert(owners[row] == (join ? first : row));
          assert(counts[row] >= 0 && counts[row] <= T);
          for (int j = 0; j < counts[row]; ++j) {
            int token = mt[row * T + j], slot = ms[row * T + j];
            assert(slot >= 0 && slot < C);
            if (join) assert(transferred.insert(token).second);
            kv[owners[row] * C + slot] = source[req[row] * N + token];
            kvLayer2[owners[row] * C + slot] = source[req[row] * N + token] + 1000000;
          }
          for (int j = 0; j < C / BS; ++j) assert(pages[row * (C / BS) + j] == owners[row] * (C / BS) + j);
        }
        // Transfer must complete for the entire group before ANY query is evaluated.
        for (int row = first; row < end; ++row)
          for (int j = 0; j < T; ++j) {
            int token = top[row * T + j], slot = cur[row * T + j];
            if (!JointTokenValid(a, row, token)) {
              assert(slot == -1);
              continue;
            }
            assert(slot >= 0 && slot < C);
            assert(st[owners[row] * C + slot] == token);
            assert(kv[owners[row] * C + slot] == source[req[row] * N + token]);
            assert(kvLayer2[owners[row] * C + slot] == source[req[row] * N + token] + 1000000);
            ++checks;
          }
        for (int row = first; row < end; ++row)
          if (row == first || !join) {
            std::set<int> slots(lru.begin() + row * C, lru.begin() + (row + 1) * C);
            assert(slots.size() == (size_t)C);
            assert(*slots.begin() == 0 && *slots.rbegin() == C - 1);
          }
        first = end;
      }
    }
  }
  assert(shared > 0 && overflow > 0);
  std::cout << "PASS token checks=" << checks << " shared runs=" << shared << " overflow runs=" << overflow << "\n";
}
