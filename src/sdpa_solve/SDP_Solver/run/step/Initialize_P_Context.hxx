#pragma once

#include "Initialize_P_Config.hxx"
#include "Vector_Block_Diagonal_Matrix_Residues_Window.hxx"
#include "bigint_trmm/blas_jobs/create_blas_job_schedule.hxx"
#include "sdpb_util/Timers/Timers.hxx"
#include "sdpb_util/bigint_shared_memory/blas_jobs/Blas_Job_Schedule.hxx"
#include "sdpb_util/bigint_shared_memory/fmpz/blas_conversions.hxx"

namespace Sdpb::Sdpa
{
  struct Initialize_P_Context
  {
    Initialize_P_Config cfg;
    El::mpi::Comm shared_memory_comm;
    Fmpz_Comb comb;
    Vector_Matrix_Residues_Window<double> L_X_inv_residues;
    Vector_Matrix_Residues_Window<double> L_Y_residues;

  private:
    Shared_Window_Array_View<double> G_window_view;
    // lab (bfss #68): staging for LAB_RESTORE_CONTIGUOUS=1, serialized BigFloats
    // indexed like the elements of one prime in the G residues window.
    std::unique_ptr<Shared_Window_Array<El::byte>> restore_staging;
    // Node-collective: (re)allocates the staging window if it is too small.
    El::byte *staging_for(size_t num_elements);
    // lab (bfss #77): BLAS residues/CRT on the primes of comb (created on
    // first use)
    std::unique_ptr<Lab_Blas_Conversions::Context> blas_conversions_context;
    const Lab_Blas_Conversions::Context &blas_context();
    std::map<std::tuple<size_t, El::VerticalOrHorizontal>,
             Vector_Block_Diagonal_Matrix_Residues_Window<double>>
      G_window_cache{};

  public:
    explicit Initialize_P_Context(const Initialize_P_Config &cfg);

    Vector_Block_Diagonal_Matrix_Residues_Window<double> &
    G_residues(size_t num_primal_indices,
               El::VerticalOrHorizontal vertical_or_horizontal);

    void compute_residues(const Block_Diagonal_Matrix &bdm,
                          Vector_Matrix_Residues_Window<double> &window,
                          El::Matrix<int32_t> &block_timings_ms);
    void compute_residues(
      const std::vector<Block_Diagonal_Matrix> &bdms,
      Vector_Block_Diagonal_Matrix_Residues_Window<double> &window,
      El::Matrix<int32_t> &block_timings_ms);

    // Compute trmm for residues and restore the result from residues
    void trmm(El::LeftOrRight side, El::UpperOrLower uplo,
              El::Orientation orientation, El::UnitOrNonUnit diag,
              Vector_Matrix_Residues_Window<double> &L_window,
              Vector_Block_Diagonal_Matrix_Residues_Window<double> &G_window,
              std::vector<Block_Diagonal_Matrix> &G, Verbosity verbosity,
              Timers &timers, El::Matrix<int32_t> &block_timings_ms);

  private:
    Vector_Matrix_Residues_Window<double> create_L_window() const;

    void
    do_blas_jobs(const Blas_Job_Schedule<Trmm::Blas_Job> &schedule,
                 Vector_Matrix_Residues_Window<double> &L_window,
                 Vector_Block_Diagonal_Matrix_Residues_Window<double> &G_window,
                 Timers &timers, El::Matrix<int32_t> &block_timings_ms);

    void compute_residues_contiguous(
      const std::vector<Block_Diagonal_Matrix> &bdms,
      Vector_Block_Diagonal_Matrix_Residues_Window<double> &window,
      El::Matrix<int32_t> &block_timings_ms);

    void restore_from_residues(
      Vector_Block_Diagonal_Matrix_Residues_Window<double> &bdms_residues,
      std::vector<Block_Diagonal_Matrix> &bdms, Timers &timers,
      El::Matrix<int32_t> &block_timings_ms);
  };
}