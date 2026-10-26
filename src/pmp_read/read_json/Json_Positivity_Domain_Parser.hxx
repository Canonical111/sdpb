#pragma once

#include "pmp/Positivity_Domain.hxx"
#include "sdpb_util/assert.hxx"
#include "sdpb_util/json/Abstract_Json_Object_Parser.hxx"
#include "sdpb_util/json/Json_Float_Parser.hxx"
#include "sdpb_util/json/Json_String_Parser.hxx"
#include "sdpb_util/json/Json_Vector_Parser.hxx"

// "domain":
// {
//   "type": "object",
//   "properties":
//   {
//     "interval":
//     {
//       "type": "array",
//       "items": { "type": "string" },
//       "minItems": 2,
//       "maxItems": 2
//     },
//     "polynomialBasis": { "enum": ["monomial", "chebyshev"] }
//   }
// }
//
// {"interval": ["a", "b"]} means a <= x <= b.
// An empty object, like a missing "domain", means x >= 0.
// "polynomialBasis": "chebyshev" (interval only): the coefficients of the
// polynomials and bilinear bases are in T_i((2x - a - b)/(b - a)).
class Json_Positivity_Domain_Parser final
    : public Abstract_Json_Object_Parser<Positivity_Domain>
{
private:
  Positivity_Domain result;

  Json_Vector_Parser<Json_BigFloat_Parser> interval_parser;
  Json_String_Parser polynomial_basis_parser;

public:
  Json_Positivity_Domain_Parser(
    const bool skip,
    const std::function<void(Positivity_Domain &&)> &on_parsed,
    const std::function<void()> &on_skipped = [] {})
      : Abstract_Json_Object_Parser(skip, on_parsed, on_skipped),
        interval_parser(skip, [this](std::vector<El::BigFloat> &&value) {
          if(value.size() != 2)
            RUNTIME_ERROR("domain: \"interval\" must have two elements [a, "
                          "b], found ",
                          value.size());
          this->result.interval.emplace(value.at(0), value.at(1));
        }),
        polynomial_basis_parser(skip, [this](std::string &&value) {
          if(value == "monomial")
            this->result.polynomial_basis = Polynomial_Basis::monomial;
          else if(value == "chebyshev")
            this->result.polynomial_basis = Polynomial_Basis::chebyshev;
          else
            RUNTIME_ERROR("domain: \"polynomialBasis\" must be \"monomial\" "
                          "or \"chebyshev\", found \"",
                          value, "\"");
        })
  {}

protected:
  Abstract_Json_Reader_Handler &element_parser(const std::string &key) override
  {
    if(key == "interval")
      return interval_parser;
    if(key == "polynomialBasis")
      return polynomial_basis_parser;
    RUNTIME_ERROR("Json_Positivity_Domain_Parser: unknown key=", key);
  }

public:
  value_type get_result() override
  {
    result.validate();
    return std::move(result);
  }
  void clear_result() override { result = Positivity_Domain{}; }
  void reset_element_parsers(const bool skip) override
  {
    interval_parser.reset(skip);
    polynomial_basis_parser.reset(skip);
  }
};
