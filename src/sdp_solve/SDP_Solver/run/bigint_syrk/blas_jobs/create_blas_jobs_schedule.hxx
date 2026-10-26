#pragma once

#include <functional>

#include "Blas_Job.hxx"
#include "Blas_Job_Schedule.hxx"
#include "sdpb_util/Verbosity.hxx"

// Create BLAS jobs for matrix multiplication C = A^T B.
// Each job calculates a submatrix C_IJ = A_I^T B_J modulo some prime
// Jobs are distributed across num_ranks ranks on a node.
//
// C is a (output_matrix_height x output_matrix_width) matrix.
// If kind == syrk, C is a symmetric square matrix,
// and jobs are created only for the upper half (I<=J).
// job_cost (optional): the cost of each job for the LPT scheduling and the
// choice of the split factor, instead of its number of output elements
// (sdpb --compressLinvB: the work on the nonzero column runs)
Blas_Job_Schedule create_weighted_blas_job_schedule(
  Blas_Job::Kind kind, El::UpperOrLowerNS::UpperOrLower uplo, size_t num_ranks,
  size_t num_primes, El::Int output_matrix_height, El::Int output_matrix_width,
  Verbosity verbosity,
  const std::function<Blas_Job::Cost(const Blas_Job &)> &job_cost);

Blas_Job_Schedule
create_blas_job_schedule(Blas_Job::Kind kind,
                         El::UpperOrLowerNS::UpperOrLower uplo,
                         size_t num_ranks, size_t num_primes,
                         El::Int output_matrix_height,
                         El::Int output_matrix_width, Verbosity verbosity);
