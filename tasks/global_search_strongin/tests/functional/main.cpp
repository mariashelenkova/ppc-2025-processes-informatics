#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
#include <tuple>

#include "global_search_strongin/common/include/common.hpp"
#include "global_search_strongin/mpi/include/ops_mpi.hpp"
#include "global_search_strongin/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace global_search_strongin {

using TestType = std::tuple<int, double, double, double, int, std::function<double(double)>, double>;

class StronginFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  StronginFuncTests() : input(), expected_result(0.0) {}

  static std::string PrintTestParam(const TestType &test_param) {
    std::string result = "test_" + std::to_string(std::get<0>(test_param)) + "_from_" +
                         std::to_string(std::get<1>(test_param)) + "_to_" + std::to_string(std::get<2>(test_param)) +
                         "_epsilon_" + std::to_string(std::get<3>(test_param)) + "_max_iters_" +
                         std::to_string(std::get<4>(test_param));
    std::ranges::replace(result, '.', '_');
    std::ranges::replace(result, '-', 'm');
    return result;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const double left = std::get<1>(params);
    const double right = std::get<2>(params);
    const double epsilon = std::get<3>(params);
    const int max_iters = std::get<4>(params);
    const auto function = std::get<5>(params);

    expected_result = std::get<6>(params);
    input = std::make_tuple(left, right, epsilon, max_iters, function);
  }

  InType GetTestInputData() final {
    return input;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr double kTolerance = 1e-2;
    return std::abs(output_data - expected_result) <= kTolerance;
  }

  InType input;
  OutType expected_result;
};

namespace {

const std::array<TestType, 3> kTestParams = {
    std::make_tuple(1, -5.0, 5.0, 0.1, 500, [](double x) { return x * x; }, 0.0),
    std::make_tuple(2, 2.0, 14.0, 0.1, 500, [](double x) { return (x - 2) * (x - 2); }, 0.0),
    std::make_tuple(3, 0.0, 8.0, 0.1, 1000, [](double x) { return std::sin(x); }, -1.0),
};

const auto kTaskList =
    std::tuple_cat(ppc::util::AddFuncTask<StronginSearchSeq, InType>(kTestParams, PPC_SETTINGS_global_search_strongin),
                   ppc::util::AddFuncTask<StronginSearchMpi, InType>(kTestParams, PPC_SETTINGS_global_search_strongin));

const auto kGTestValues = ppc::util::ExpandToValues(kTaskList);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(Strongin, StronginFuncTests, kGTestValues,
                         StronginFuncTests::PrintFuncTestName<StronginFuncTests>);

TEST_P(StronginFuncTests, Runs) {
  ExecuteTest(GetParam());
}

}  // namespace

}  // namespace global_search_strongin
