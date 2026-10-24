# Multimodular per-block Cholesky and Trsm

lab (agent), dsd-lab-playground/bootstrap-speed#145. Experimental, behind `SDPB_MULTIMODULAR_TRSM=1`.

## What it replaces

In `initialize_schur_off_diagonal()` (`compute_Q.cxx`), each Schur block is factored, S_j = L_j L_j^T
(`El::Cholesky`), and then P_j = L_j^{-1} B_j is solved (`El::Trsm`, with B_j = free_var_matrix block, P_j x N).
Both run in BigFloat. For the nmax10 stress-tensor SDP on one Expanse node (128 ranks) they took 69 of 114 s per
iteration on the busiest ranks: about 300 ns per 1024-bit multiply-add, for ~P^2 N/2 of them.

## Idea

Do the O(n^3) work as matrix multiplications on integers, exactly as `bigint_syrk` does for Q = P^T P. Blocked
Cholesky and Trsm consist almost entirely of updates C -= A op(B). Only small diagonal blocks (≤ 32) need
divisions and square roots, and those stay in BigFloat.

`multimodular_gemm_sub(C, A, op(B))`:

1. **Fixed point.** Row i of A and column j of op(B) are multiplied by 2^(bits - e), where e is the exponent of
   their largest entry and bits = precision + 64 guard bits. The results are truncated to integers a_il, b_lj
   (|a|, |b| < 2^bits). The update error is then ~k 2^-bits max_l|A_il| max_l|B_lj|. For the Cholesky this is
   the usual normwise backward error |dA_ij| ≲ ε sqrt(A_ii A_jj).
2. **Residues** modulo primes p < 2^20.7 with k p^2 < 2^53 (as FLINT's mul_blas), with product M > 8 max|c|.
   Each integer is split into 24-bit chunks c_t, and the residues of a batch are one dgemm:
   |x| mod p = (Σ_t c_t · (2^(24t) mod p)) mod p, exact in double since T 2^24 p < 2^53.
   The residues are stored as int32 (half the memory of double) and converted to double per prime before the
   dgemm. The residues of A are multiplied by u_p = (M/p)^{-1} mod p, which saves that step in the CRT.
3. **One `cblas_dgemm` per prime.**
4. **CRT by BLAS.** With y_p = u_p c mod p in [0, p):
   c = Σ_p y_p (M/p) − q M, with q = round(Σ_p y_p / p) computed in double (|c|/M < 1/8 leaves room for its
   rounding error). M/p is split into 24-bit chunks m_pt. S_t = Σ_p y_p m_pt is exact in double, and one dgemm
   gives all S_t. Only chunks t ≥ t0 are formed: the dropped low part is < 2^(bits−64), below the
   fixed-point truncation of step 1. FLINT's exact `fmpz_multi_CRT_ui` is kept behind `config.flint_crt`, and
   both give bitwise-identical results on the tested blocks.
5. C_ij -= c_ij 2^(e_i + e_j − 2 bits).

`multimodular_cholesky_lower`: recursive halving; A11 → chol, L21 = A21 L11^{-T}, A22 −= L21 L21^T (lower part only).

`multimodular_trsm_left_lower`: two variants.
- Recursive halving: X1 = L11^{-1} B1; B2 −= L21 X1; X2 = L22^{-1} B2.
- Left-looking over panels of 32 rows: B_p −= L_{p,<p} X_{<p}, then X_p = L_pp^{-1} B_p. Each entry of L and
  X, and each output, is converted once. X's residues are stored at a fixed scale per column with 32 bits of
  headroom; if a later panel exceeds it, that column's residues are recomputed.
- Left-looking is used when B has ≤ 2x as many columns as L has rows.

## In SDPB (`compute_Q.cxx`)

A block of size ≥ 64 lives on a grid of g ranks.
- Its lower triangle is held once per node, in an MPI-3 shared-memory window (`Shared_Lower_Matrix`). Each rank
  has its own El::Matrix of mpf structs pointing into the window. Headers (size, exponent) are synced after
  each write phase.
- The ranks write their own entries of S into the window.
- The Cholesky splits, at each recursion level of ≥ 256 rows, the L21 solve and the A22 update by columns over
  the grid. Each rank writes its columns in place; smaller parts are computed by grid rank 0.
- The Trsm splits B's columns over the grid ([STAR,VR]). Each rank solves its columns with the shared L.
- The factor's strictly upper triangle keeps the input values, as El::Cholesky leaves it.

If a grid spans several nodes (or with `SDPB_MULTIMODULAR_SHARED=0`), every rank holds its own copy of the lower
triangle instead, filled by panel allgathers.

## Settings (environment variables, read once)

| variable | default | meaning |
|---|---|---|
| `SDPB_MULTIMODULAR_TRSM` | 0 | 1 enables the multimodular path |
| `SDPB_MULTIMODULAR_BASE` | 32 | diagonal block size in BigFloat; panel height of the left-looking Trsm |
| `SDPB_MULTIMODULAR_MIN_SIZE` | 2·base | smaller blocks use El |
| `SDPB_MULTIMODULAR_DIST_MIN` | 256 | Cholesky levels of at least this size are split over the grid |
| `SDPB_MULTIMODULAR_GUARD` | 64 | guard bits of the fixed point |
| `SDPB_MULTIMODULAR_CHUNK` | 256 | columns of op(B) per batch in the GEMM |
| `SDPB_MULTIMODULAR_LL_MB` | 384 | memory for the left-looking Trsm's stored residues |
| `SDPB_MULTIMODULAR_LL_RATIO` | 2 | left-looking if columns ≤ ratio · rows |
| `SDPB_MULTIMODULAR_LEFT_LOOKING` | 1 | 0: always recursive |
| `SDPB_MULTIMODULAR_GEMM_A_MB` | 0 (off) | bound on A's residues (row blocks) |
| `SDPB_MULTIMODULAR_FLINT_CRT` | 0 | exact FLINT CRT (for checks) |
| `SDPB_MULTIMODULAR_SHARED` | 1 | shared-memory factor |

Also in `compute_Q.cxx`, `SDPB_DUMP_BLOCK=<b>` and `SDPB_DUMP_DIR` (optionally `SDPB_DUMP_ITER`) write one
block's Schur complement and B for `mm_trsm_bench` (`src/mm_trsm_bench`).

## Measurements

See dsd-lab-playground/bootstrap-speed `calc/sdpb_mm_trsm/REPORT.md`. In short, for the nmax10 SDP:
- Kernel, block 975 x 1834, one core: 10.5x faster than El, and at least as accurate against a 1536-bit
  reference.
- One node, 128 ranks: 58 s/iteration vs 113.5 s, with iterations 1-3 equal to production in every printed
  column.
