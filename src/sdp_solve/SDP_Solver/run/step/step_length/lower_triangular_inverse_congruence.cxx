#include "sdp_solve/Block_Diagonal_Matrix.hxx"

#include <cstdlib>
#include <stdexcept>
#include <string>

// Full-working-precision prototype (#49), independent of the retired double path.
// Both modes keep BigFloat throughout; only the lower triangle is consumed.
// SDPB_HIGH_PRECISION_CONGRUENCE=trsm (default) | twosided.
// A := L^{-1} A L^{-T}
void lower_triangular_inverse_congruence(const Block_Diagonal_Matrix &L,
                                         Block_Diagonal_Matrix &A)
{
  static const std::string mode = [] {
    const char *value = std::getenv("SDPB_HIGH_PRECISION_CONGRUENCE");
    const std::string result = value ? value : "trsm";
    if(result != "trsm" && result != "twosided")
      throw std::invalid_argument("Unknown SDPB_HIGH_PRECISION_CONGRUENCE: " + result);
    return result;
  }();
  for(size_t b = 0; b < A.blocks.size(); b++)
    {
      if(mode == "twosided")
        {
          // TwoSidedTrsm computes L^-1 A L^-T using symmetry. Elemental
          // instantiates both local and AbstractDistMatrix APIs for BigFloat.
          if(A.blocks[b].Grid().Size() == 1)
            El::TwoSidedTrsm(El::UpperOrLowerNS::LOWER,
                             El::UnitOrNonUnit::NON_UNIT,
                             A.blocks[b].Matrix(), L.blocks[b].LockedMatrix());
          else
            El::TwoSidedTrsm(El::UpperOrLowerNS::LOWER,
                             El::UnitOrNonUnit::NON_UNIT,
                             A.blocks[b], L.blocks[b]);
          continue;
        }
      El::Trsm(El::LeftOrRight::RIGHT, El::UpperOrLowerNS::LOWER,
               El::Orientation::TRANSPOSE, El::UnitOrNonUnit::NON_UNIT,
               El::BigFloat(1), L.blocks[b], A.blocks[b]);
      El::Trsm(El::LeftOrRight::LEFT, El::UpperOrLowerNS::LOWER,
               El::Orientation::NORMAL, El::UnitOrNonUnit::NON_UNIT,
               El::BigFloat(1), L.blocks[b], A.blocks[b]);
    }
}
