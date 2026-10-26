#include "sdp_solve/SDP_Solver.hxx"
#include "sdpb_util/assert.hxx"
#include "sdpb_util/copy_matrix.hxx"

#include <functional>
#include <optional>

// Solve the Schur complement equation for dx, dy.
//
// - As inputs, dx and dy are the residues r_x and r_y on the
//   right-hand side of the Schur complement equation.
// - As outputs, dx and dy are overwritten with the solutions of the
//   Schur complement equation.
//
// The equation is solved using the block-decomposition described in
// the manual.
//
// #184: with streaming (SDPB_STREAM_P) schur_off_diagonal = L^{-1} B is not stored
// (empty), and its two products are formed from B and L instead:
//   (L^{-1} B)^T u = B^T (L^{-T} u),   (L^{-1} B) w = L^{-1} (B w).
// off_diagonal(block): block of L^{-1} B, or of B when streaming.
static void solve_schur_complement_equation(
  const Block_Diagonal_Matrix &schur_complement_cholesky, const bool streaming,
  const std::function<const El::DistMatrix<El::BigFloat> &(size_t)>
    &off_diagonal,
  const El::DistMatrix<El::BigFloat> &Q, Block_Vector &dx, Block_Vector &dy)
{
  // dx = schur_complement^{-1}.dx
  lower_triangular_solve(schur_complement_cholesky, dx);
  // Streaming: v = L^{-T} dx, so that B^T v = (L^{-1} B)^T dx.
  std::optional<Block_Vector> v;
  if(streaming)
    {
      v.emplace(dx);
      lower_triangular_transpose_solve(schur_complement_cholesky, *v);
    }

  El::DistMatrix<El::BigFloat> dy_dist;
  Zeros(dy_dist, Q.Height(), 1);
  {
    El::Matrix<El::BigFloat> dy_sum;
    Zeros(dy_sum, Q.Height(), 1);

    for(size_t block = 0; block < dx.blocks.size(); ++block)
      {
        Gemv(El::OrientationNS::TRANSPOSE, El::BigFloat(-1),
             off_diagonal(block),
             streaming ? v->blocks[block] : dx.blocks[block], El::BigFloat(1),
             dy.blocks[block]);

        // Locally sum contributions to dy
        for(int64_t row = 0; row < dy.blocks[block].LocalHeight(); ++row)
          {
            int64_t global_row(dy.blocks[block].GlobalRow(row));
            for(int64_t column = 0; column < dy.blocks[block].LocalWidth();
                ++column)
              {
                int64_t global_column(dy.blocks[block].GlobalCol(column));
                dy_sum(global_row, global_column)
                  += dy.blocks[block].GetLocal(row, column);
              }
          }
      }

    // Send out updates for dy
    El::BigFloat zero(0);
    for(int64_t row = 0; row < dy_sum.Height(); ++row)
      for(int64_t column = 0; column < dy_sum.Width(); ++column)
        {
          if(dy_sum(row, column) != zero)
            {
              dy_dist.QueueUpdate(row, column, dy_sum(row, column));
            }
        }
  }
  dy_dist.ProcessQueues();

  // dy_dist = Q^{-1}.dy_dist
  El::cholesky::SolveAfter(El::UpperOrLowerNS::UPPER,
                           El::OrientationNS::NORMAL, Q, dy_dist);
  El::DistMatrix<El::BigFloat, El::STAR, El::STAR> dy_local(dy_dist);

  // dx += schur_off_diagonal.dy
  // (streaming: v = B.dy, v = L^{-1} v, dx += v)
  for(size_t block = 0; block < dx.blocks.size(); ++block)
    {
      copy_matrix(dy_local, dy.blocks[block]);
      Gemv(El::OrientationNS::NORMAL, El::BigFloat(1),
           off_diagonal(block), dy.blocks[block],
           El::BigFloat(streaming ? 0 : 1),
           streaming ? v->blocks[block] : dx.blocks[block]);
    }
  if(streaming)
    {
      lower_triangular_solve(schur_complement_cholesky, *v);
      for(size_t block = 0; block < dx.blocks.size(); ++block)
        El::Axpy(El::BigFloat(1), v->blocks[block], dx.blocks[block]);
    }

  // dx = schur_complement^{-T}.dx
  lower_triangular_transpose_solve(schur_complement_cholesky, dx);
}

void solve_schur_complement_equation(
  const Block_Diagonal_Matrix &schur_complement_cholesky,
  const Block_Matrix &schur_off_diagonal, const SDP &sdp,
  const El::DistMatrix<El::BigFloat> &Q, Block_Vector &dx, Block_Vector &dy)
{
  const bool streaming = schur_off_diagonal.blocks.empty();
  El::DistMatrix<El::BigFloat> B_scratch; // for --sparseB
  solve_schur_complement_equation(
    schur_complement_cholesky, streaming,
    [&](const size_t block) -> const El::DistMatrix<El::BigFloat> & {
      return streaming ? sdp.free_var_block(block, B_scratch)
                       : schur_off_diagonal.blocks.at(block);
    },
    Q, dx, dy);
}

// For callers that always store L^{-1} B (approx_objective, whose syrk config never
// streams).
void solve_schur_complement_equation(
  const Block_Diagonal_Matrix &schur_complement_cholesky,
  const Block_Matrix &schur_off_diagonal,
  const El::DistMatrix<El::BigFloat> &Q, Block_Vector &dx, Block_Vector &dy)
{
  ASSERT(!schur_off_diagonal.blocks.empty(),
         "schur_off_diagonal is empty: streaming needs B");
  solve_schur_complement_equation(
    schur_complement_cholesky, false,
    [&](const size_t block) -> const El::DistMatrix<El::BigFloat> & {
      return schur_off_diagonal.blocks.at(block);
    },
    Q, dx, dy);
}
