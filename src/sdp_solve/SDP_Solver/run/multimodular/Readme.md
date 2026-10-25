# Multimodular per-block Cholesky and Trsm

Enabled by `sdpb --multimodularTrsm` (default: off).

## What it replaces

In `initialize_schur_off_diagonal()` (`step/initialize_schur_complement_solver/compute_Q.cxx`), each Schur block is
factored, S_j = L_j L_j^T (`El::Cholesky`), and then P_j = L_j^{-1} B_j is solved (`El::Trsm`, with B_j the block of
`free_var_matrix`, P_j x N). Both run in BigFloat. For a large SDP (nmax=10 stress-tensor bootstrap, 1024 bits, one
128-core node) they take more than half of each iteration: about 300 ns per 1024-bit multiply-add, for ~P^2 N/2 of
them.

## Idea

Do the O(n^3) work as matrix multiplications on integers, as `bigint_syrk` does for Q = P^T P (see
`../bigint_syrk/Readme.md`). Blocked Cholesky and Trsm consist almost entirely of updates C -= A op(B). Only small
diagonal blocks (at most 32 x 32) need divisions and square roots, and those stay in BigFloat.

`multimodular_gemm_sub(C, A, op(B))` (`multimodular_gemm.cxx`):

1. **Fixed point.** Row i of A and column j of op(B) are multiplied by 2^(bits - e), where e is the exponent of
   their largest entry and bits = precision + 64 guard bits. The results are truncated to integers a_il, b_lj
   (|a|, |b| < 2^bits). The update error is then ~k 2^-bits max_l|A_il| max_l|B_lj|. For the Cholesky this is
   the usual normwise backward error |dA_ij| ≲ ε sqrt(A_ii A_jj).
   The scale factors are powers of 2, so the scaling itself is exact. `Matrix_Normalizer` (used by `bigint_syrk`)
   is not used here: it normalizes columns only, by their 2-norms, at exactly `precision` bits. An entry 2^-k below
   its row's norm then keeps only precision - k bits, which costs accuracy in a triangular solve.
2. **Residues** modulo primes p < 2^21 from `Fmpz_Comb`, computed directly from the mpf limbs with BLAS
   (`../bigint_syrk/fmpz/blas_residues.hxx`): each integer is split into 24-bit chunks, and the residues of a batch
   are one dgemm with the table 2^(24t) mod p. They are stored as int32 and converted to double per prime.
3. **One `cblas_dgemm` per prime.**
4. **CRT, also by BLAS** (`blas_residues.hxx`): one dgemm of the residues with the 24-bit chunks of M/p, then the
   chunks are added up with carries. Chunks below 2^(24 t0) are not formed; the error this leaves is ~2^89 below
   the fixed-point truncation of step 1. t0 is a multiple of 8 (whole limbs), so the results agree bit for bit
   with FLINT's exact `fmpz_multi_CRT_ui` (`Multimodular_Blas_Config::flint_crt`, used by the unit tests).
   On the same data and in the same loop, the restore step (CRT, conversion to BigFloat, update of C) takes 2.3 s
   instead of 6.1 s with FLINT's CRT (975 x 975 Cholesky + Trsm with 1834 columns, 1024 bits, one core).
5. C_ij -= c_ij 2^(e_i + e_j − 2 bits).

`multimodular_cholesky_lower` (`blocked_cholesky_trsm.cxx`): recursive halving; A11 → chol, L21 = A21 L11^{-T},
A22 −= L21 L21^T (lower part only).

`multimodular_trsm_left_lower`: two variants.
- Recursive halving: X1 = L11^{-1} B1; B2 −= L21 X1; X2 = L22^{-1} B2.
- Left-looking over panels of 32 rows: B_p −= L_{p,<p} X_{<p}, then X_p = L_pp^{-1} B_p. Each entry of L and
  X, and each output, is converted once. X's residues are stored at a fixed scale per column with 32 bits of
  headroom; if a later panel exceeds it, that column's residues are recomputed.
- Left-looking is used when B has at most 2x as many columns as L has rows.

## In SDPB (`Multimodular_Block_Factor`)

A block lives on a grid of g ranks.
- If the grid is on one node, the lower triangle of its factor is held once per node, in an MPI-3 shared-memory
  window (`Shared_Window_Array`). Each rank has its own El::Matrix of mpf structs whose limbs point into the window;
  sizes and exponents are published to the window after each write phase (`Fence()`).
  The ranks write their own entries of S into the window. The Cholesky splits, at each recursion level of at least
  256 rows, the L21 solve and the A22 update by columns over the grid; each rank writes its columns in place, and
  smaller parts are computed by grid rank 0. The Trsm copies B into a second shared window and splits its columns
  over the grid; each rank solves its columns in place with the shared L.
- If the grid spans several nodes, every rank holds its own copy of the lower triangle instead, filled by panel
  allgathers, and B's columns are split over the grid with a [STAR,VR] copy.
- On one rank, B is solved in place.
- The factor's strictly upper triangle keeps the input values, as `El::Cholesky` leaves it.

Results do not depend on which of these layouts is used (unit test `Multimodular_Block_Factor`).

## Settings

The defaults are in `Multimodular_Blas_Config` (`multimodular_blas.hxx`); there are no run-time options besides
`--multimodularTrsm`.

| setting | default | meaning |
|---|---|---|
| `base_size` | 32 | diagonal block size in BigFloat; panel height of the left-looking Trsm |
| `guard_bits` | 64 | guard bits of the fixed point (see 1.) |
| `column_chunk` | 256 | columns of op(B) per batch in the GEMM (memory for residues) |
| `left_looking` | true | left-looking Trsm for few columns; recursive otherwise |
| `left_looking_max_ratio` | 2 | left-looking if columns ≤ ratio · rows |
| `growth_bits` | 32 | left-looking Trsm: headroom of the per-column scale of X |
| `left_looking_bytes` | 384 MiB | left-looking Trsm: memory for the stored residues of X |
| `gemm_A_bytes` | 0 (off) | bound on the memory for A's residues (row blocks) |
| `flint_crt` | false | exact FLINT CRT instead of the BLAS CRT (tests) |
| `Multimodular_Block_Factor::distributed_min_size` | 256 | Cholesky levels of at least this size are split over the grid |

## Tests

`test/src/unit_tests/cases/multimodular_blas.test.cxx`:
- BLAS residues equal FLINT's (`fmpz_multi_mod_uint32_stride`); the BLAS CRT is exact for t0 = 0 and within its
  error bound otherwise, against FLINT's exact CRT;
- GEMM, Cholesky and both Trsm variants against El at the working precision (backward errors), and BLAS CRT vs
  FLINT CRT bit for bit;
- `Multimodular_Block_Factor` on distributed blocks (one rank, all ranks; shared window and per-rank copies)
  against `El::Cholesky` + `El::Trsm`.

## Measurements

nmax=10 stress-tensor SDP, 1024 bits (dsd-lab-playground/bootstrap-speed `calc/sdpb_mm_trsm`,
`calc/sdpb_mm_vs_sdpa_trmm`, `calc/sdpb_mm_pr`):
- Kernel, block 975 x 1834, one core: Cholesky + Trsm 35 s vs El's 362 s. Backward errors at El's level (2^-1021)
  against a 2048-bit MPFR reference, also at late iterations with κ(S) ~ 2^330.
- One node, 128 ranks (`srun`, `--maxSharedMemory=16G`), same node: 54.0 s/iteration vs El's 107.5 s, iterations
  equal to El's in every printed column.
