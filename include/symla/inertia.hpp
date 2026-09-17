#pragma once

// A tiny standalone header for symla::Inertia, split out of solver.hpp so
// that dense_kernel.hpp (which only needs the Inertia type) does not have
// to include the full solver.hpp -- which, starting with Phase 3, itself
// includes multifrontal.hpp -> dense_kernel.hpp, and that cycle back would
// otherwise break translation units that include dense_kernel.hpp directly
// (e.g. test/unit/dense_kernel_test.cpp): dense_kernel.hpp's own
// #pragma once would cause the re-entrant include from multifrontal.hpp to
// be skipped mid-file, before DenseLDLT was actually defined.

namespace symla {

struct Inertia {
  int n_pos = 0;
  int n_neg = 0;
  int n_zero = 0;
};

}  // namespace symla
