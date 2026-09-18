#pragma once

// Phase 6: generic OpenMP-task DAG driver for tree-shaped dependency
// graphs, factored out of multifrontal.hpp so the traversal/scheduling
// policy is independent of what a "node" actually is (in this project,
// always a supernode, but keeping it generic makes the recursive-task
// pattern easy to unit-reason about on its own).
//
// The dependency direction this implements is "process a node only after
// all of its children have been processed" -- exactly the multifrontal
// elimination-tree requirement (a supernode's extend-add assembly needs
// every child's already-finalized generated element/update matrix). This
// is the standard tree/DAG task-parallel pattern described in the open
// literature for multicore sparse Cholesky/LDL^T (Hogg, Reid, Scott 2010,
// "Design of a Multicore Sparse Cholesky Factorization Using DAGs", SIAM
// J. Sci. Comput.; the PaStiX line of work by Hénon/Ramet/Roman) --
// implemented here from that general public knowledge, not from any
// proprietary MA57/HSL source.
//
// Usage: called from inside an active `#pragma omp parallel` region with
// exactly one thread driving the top-level call (typically from
// `#pragma omp single`); internally it recursively spawns `#pragma omp
// task` for children and `#pragma omp taskwait`s before invoking `process`
// on a node, so the OpenMP runtime is free to schedule sibling subtrees
// (and, near the leaves, sibling leaves) onto different worker threads.
// Without `SYMLA_HAVE_OPENMP` (or if `SYMLA_HAVE_OPENMP` is defined but the
// caller never opens a parallel region), this degrades gracefully to a
// plain sequential postorder recursion -- correct, just not parallel.

#include <functional>
#include <vector>

namespace symla {
namespace parallel {

// Runs `process(node)` for every node reachable from `roots` via
// `children[node]`, guaranteeing all of a node's children (transitively)
// are processed before `process(node)` runs.
//
// `subtreeWeight[node]` is a scheduling-only cost proxy for the subtree
// rooted at `node` (e.g. subtree size in nodes, or a FLOP estimate) --
// children whose subtree weight is below `taskCutoff` are executed inline
// by the current thread (an "undeferred" call, not spawned as a separate
// `omp task`) rather than paying task-creation overhead for work that is
// too small to be worth distributing; children at/above the cutoff are
// spawned as tasks so the runtime can run them concurrently with their
// siblings. `taskCutoff <= 0` spawns a task for every non-empty subtree.
template <typename ProcessFn>
inline void runTaskDag(const std::vector<int>& roots, const std::vector<std::vector<int>>& children,
                        const std::vector<long long>& subtreeWeight, long long taskCutoff, ProcessFn&& process) {
#ifdef SYMLA_HAVE_OPENMP
  std::function<void(int)> visit = [&](int node) {
    // `omp taskwait` itself is not free (a real synchronization point with
    // the runtime, even with zero outstanding child tasks) -- with many
    // thousands of tree nodes, unconditionally paying it at every single
    // one adds up fast once the cutoff below is doing its job and most
    // subtrees are executed inline (no tasks spawned for them at all). Only
    // pay for it when this node actually spawned at least one child task.
    bool spawnedAny = false;
    for (int c : children[node]) {
      if (subtreeWeight[c] >= taskCutoff) {
        spawnedAny = true;
#pragma omp task default(shared) firstprivate(c)
        visit(c);
      } else {
        // Too small to be worth the task-creation overhead: run inline,
        // still fully sequentially-consistent w.r.t. `node`'s own
        // processing below since we're still on the same call stack.
        visit(c);
      }
    }
    if (spawnedAny) {
#pragma omp taskwait
    }
    process(node);
  };
  for (int r : roots) {
#pragma omp task default(shared) firstprivate(r)
    visit(r);
  }
#pragma omp taskwait
#else
  std::function<void(int)> visit = [&](int node) {
    for (int c : children[node]) visit(c);
    process(node);
  };
  for (int r : roots) visit(r);
#endif
}

}  // namespace parallel
}  // namespace symla
