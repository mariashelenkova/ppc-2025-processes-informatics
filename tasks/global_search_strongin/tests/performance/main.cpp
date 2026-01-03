#include <gtest/gtest.h>

#include <cmath>
#include <tuple>

#include "global_search_strongin/common/include/common.hpp"
#include "global_search_strongin/mpi/include/ops_mpi.hpp"
#include "global_search_strongin/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace global_search_strongin {

class StronginPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  StronginPerfTests() : input_(), expected_result_(0.0) {}

 protected:
  void SetUp() override {
    const double left = -5.0;
    const double right = 5.0;
    const double epsilon = 1e-4;
    const int max_iters = 40000;
    input_ = std::make_tuple(left, right, epsilon, max_iters, Function);
    expected_result_ = -6.1;
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr double kTolerance = 1e-2;
    return std::abs(output_data - expected_result_) <= kTolerance;
  }

 private:
  InType input_;
  OutType expected_result_;

  static double Function(double x) {
    return (0.002 * x * x) + (5.0 * std::sin(30.0 * x)) + std::sin(200.0 * std::sin(50.0 * x)) +
           (0.1 * std::cos(300.0 * x));
  }
};

namespace {

const auto kPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, StronginSearchMpi, StronginSearchSeq>(PPC_SETTINGS_global_search_strongin);

const auto kGTestValues = ppc::util::TupleToGTestValues(kPerfTasks);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(StronginPerf, StronginPerfTests, kGTestValues, StronginPerfTests::CustomPerfTestName);

TEST_P(StronginPerfTests, Runs) {
  ExecuteTest(GetParam());
}

}  // namespace

}  // namespace global_search_strongin
