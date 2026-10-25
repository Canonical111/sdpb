#pragma once

#include "pmp/Positivity_Domain.hxx"
#include "sdpb_util/assert.hxx"
#include "sdpb_util/json/Abstract_Json_Object_Parser.hxx"
#include "sdpb_util/json/Json_Float_Parser.hxx"
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
//     }
//   }
// }
//
// {"interval": ["a", "b"]} means a <= x <= b.
// An empty object, like a missing "domain", means x >= 0.
class Json_Positivity_Domain_Parser final
    : public Abstract_Json_Object_Parser<Positivity_Domain>
{
private:
  Positivity_Domain result;

  Json_Vector_Parser<Json_BigFloat_Parser> interval_parser;

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
        })
  {}

protected:
  Abstract_Json_Reader_Handler &element_parser(const std::string &key) override
  {
    if(key == "interval")
      return interval_parser;
    RUNTIME_ERROR("Json_Positivity_Domain_Parser: unknown key=", key);
  }

public:
  value_type get_result() override
  {
    result.validate();
    return std::move(result);
  }
  void clear_result() override { result.interval.reset(); }
  void reset_element_parsers(const bool skip) override
  {
    interval_parser.reset(skip);
  }
};
