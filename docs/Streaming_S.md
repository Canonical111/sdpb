# Streaming S (`SDPB_STREAM_P`) and the band height (`SDPB_STREAM_P_BAND_ROWS`)

This applies to SDPA-format input (`.dat-s`) only.

## What streaming does

Each iteration forms the Schur complement S = P^T P, where column p of P is vec(L_X^{-1} F_p L_Y).
P has sum(dim_b^2) rows and m columns (m = number of constraints), and on the default path it is
usually the largest object of the solve. SDPB builds all of it as `BigFloat`, converts all of it
to residues modulo a set of primes in a shared memory window, and squares those with BLAS
(see [bigint_syrk/Readme.md](../src/sdp_solve/SDP_Solver/run/bigint_syrk/Readme.md)).

With `SDPB_STREAM_P=1` in the environment, P is never formed. SDPB produces P a band of rows at a
time, converts the band to residues, adds its contribution to S, and moves on to the next band.
Only one band exists at a time, so the memory is set by the band height rather than by the size of P.

```
export SDPB_STREAM_P=1
mpirun -n 8 sdpb --sdpDir problem.dat-s --outDir out ...
```

The results agree with the default path to working precision, but not bit for bit, because the
products are associated in a different order. On a single node, different band heights give bitwise
identical results.

## What the band height costs and buys

The band height is counted in *rows per prime*: the height of the syrk input window, summed over the
MPI groups of the node. One band takes

| | size |
|---|---|
| syrk input window (shared memory) | band rows × m × n_primes × 8 bytes |
| the band as `BigFloat`, spread over the ranks | band rows × m × `BigFloat` size |

where n_primes grows with `--precision`. For `12_min` (m = 330, 512 bits, 53 primes, 104-byte
`BigFloat`), 990 rows take a 132 MB input window and a 32 MB `BigFloat` band. Both grow linearly in m.

On a single node, taller bands buy only fewer passes. Every band is scaled by the same per-column
factors, so the residues of all bands add up in the output window, and S is restored once per
iteration however many bands there are. What remains per band is BLAS calls on shorter inputs and,
with several ranks, one synchronization of the node's ranks. On one rank, bands of about a thousand
rows were as fast as one band holding all of P. On 8 ranks they were 11% slower (table below).

Across nodes, each band is restored into S on its own, at about the cost of one m × m
reconstruction, so fewer and taller bands are clearly faster there.

## Setting the band height

On a single node, bands are at most `SDPB_STREAM_P_BAND_ROWS` rows per prime tall (default 1024):

```
export SDPB_STREAM_P=1
export SDPB_STREAM_P_BAND_ROWS=512   # shorter bands: less memory, a little slower
```

- `--maxSharedMemory` still applies: the band height is the smaller of the cap and what fits in
  `--maxSharedMemory`.
- `SDPB_STREAM_P_BAND_ROWS=0` removes the cap, and the bands are then as tall as `--maxSharedMemory`
  allows. On the SDPA path an unset `--maxSharedMemory` means no limit, so with both unset the whole
  of P is held as residues in a single band, taking about as much memory as the default path. Do not
  combine the two.
- A band holds at least one column of G_p from every block it touches, so it is never shorter than
  the largest block of an MPI group. With a cap below that, SDPB uses the shortest bands that still
  hold the block (in the BFSS run below, a cap of 256 gave bands of 449 rows).
- The cap is not applied across nodes, for the reason above. On several nodes, bound the memory with
  `--maxSharedMemory`, as without streaming.

## Checking what SDPB chose

With `--verbosity debug` (or `--verbosity 2`), the memory report at the start of the solve contains

```
    SDPB_STREAM_P band rows per prime: 990 (cap 1024, SDPB_STREAM_P_BAND_ROWS)
```

followed by the estimated sizes of the shared memory windows (`input window(s)`, `output window`)
and of the band (`P band (BigFloat)`).

`SDPB_STREAM_P_PROFILE=1` adds one line per iteration with the number of bands and the time spent
forming rows, converting them to residues, in BLAS, and in the restore:

```
S_stream: 21 bands, 1 passes; rows 8.68 s, scale 0.17 s, syrk 11.5 s (last pass: residues 7.56, clear 0.07, blas 3.46, reduce 0.39), restore 0.06 s; ...
```

## Measured

Expanse (AMD EPYC 7742), one node, 3 iterations, no memory flags unless stated. Time is the mean of
iterations 2 and 3; memory is the peak PSS summed over all ranks, sampled every second. x and Y were
bitwise equal at every band height on each problem.

| band rows (cap) | BFSS L11, 8 ranks, 768 bits | `12_min`, 1 rank, 512 bits |
|---|---|---|
| default path (no streaming) | 24.4 s, 13.71 GiB | 74.9 s, 8.00 GiB |
| no cap (0) | 12.0 s, 7.76 GiB | 29.9 s, 4.89 GiB |
| 8192 | | 30.8 s, 1.83 GiB |
| 4096 | 12.5 s, 2.98 GiB | 30.1 s, 1.25 GiB |
| 2048 | 13.2 s, 2.37 GiB | 29.6 s, 0.79 GiB |
| **1024 (default)** | **13.3 s, 2.10 GiB** | **29.3 s, 0.55 GiB** |
| 512 | 14.0 s, 1.97 GiB | 30.2 s, 0.44 GiB |
| 256 | 13.7 s, 1.95 GiB | 32.2 s, 0.38 GiB |
| `--maxSharedMemory 16M` (99 rows) | | 37.1 s, 0.31 GiB |

The bands come out at or a little below the cap (983 and 990 rows at 1024); at 256, BFSS needed
449-row bands for its largest block. Under Open MPI's UCX transport each rank maps about
100 MiB whatever SDPB does. On a single node, `OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader` avoids that:
BFSS at 2048 rows then peaked at 1.08 GiB instead of 2.37 GiB, at the same speed.

## Choosing a value

- Keep the default (1024) for single-node runs. On `12_min` (1 rank) it was the fastest height,
  using 9× less memory than without the cap. On BFSS (8 ranks) it was 1% slower than 2048 rows and
  11% slower than one band, using 3.7× less memory than one band.
- If memory is plentiful and the node runs many ranks, taller bands can be faster: BFSS ran at
  12.5 s and 2.98 GiB with 4096 rows. Only 1 and 8 ranks were measured. With many more ranks per
  node, compare the `syrk` time of the `S_stream` line at two caps.
- If memory is tight, 256 to 512 rows save a little more, at 3 to 10% in time. Below that, what is
  left is S, X, Y and their factorizations, and the process itself; one rank of `12_min` got down
  to 0.31 GiB only with a 99-row band, 26% slower.
- If m is large, check the input window size in the memory report: it grows linearly in m, and
  a lower cap shrinks it in proportion.
- On several nodes, use `--maxSharedMemory` instead.
