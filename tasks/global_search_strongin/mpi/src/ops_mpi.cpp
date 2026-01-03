#include "global_search_strongin/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "global_search_strongin/common/include/common.hpp"

namespace global_search_strongin {
namespace {

constexpr double kReliability = 2.0;

struct IntervalCharacteristic {
  double value = 0.0;
  int index = -1;
};

double Evaluate(const InType &input, double x) {
  const auto &objective = std::get<4>(input);
  return objective ? objective(x) : 0.0;
}

bool Comparator(const SamplePoint &lhs, const SamplePoint &rhs) {
  return lhs.x < rhs.x;
}

}  // namespace

StronginSearchMpi::StronginSearchMpi(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool StronginSearchMpi::ValidationImpl() {
  const auto &input = GetInput();
  const auto &objective = std::get<4>(input);
  const double left = std::get<0>(input);
  const double right = std::get<1>(input);
  const double epsilon = std::get<2>(input);
  const int max_iterations = std::get<3>(input);

  if (!objective) {
    return false;
  }
  if (!(left < right)) {
    return false;
  }
  if (epsilon <= 0.0) {
    return false;
  }
  return max_iterations > 0;
}

bool StronginSearchMpi::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  points_.clear();
  const auto &input = GetInput();

  const double left_bound = std::get<0>(input);
  const double right_bound = std::get<1>(input);
  const SamplePoint left{.x = left_bound, .value = Evaluate(input, left_bound)};
  const SamplePoint right{.x = right_bound, .value = Evaluate(input, right_bound)};
  points_.push_back(left);
  points_.push_back(right);
  if (!std::ranges::is_sorted(points_, Comparator)) {
    std::ranges::sort(points_, Comparator);
  }
  best_x_ = left.value < right.value ? left.x : right.x;
  best_value_ = std::min(left.value, right.value);
  iterations_done_ = 0;
  return true;
}

bool StronginSearchMpi::RunImpl() {
  const auto &input = GetInput();

  const double left_bound = std::get<0>(input);
  const double right_bound = std::get<1>(input);
  const double epsilon = std::get<2>(input);
  const int max_iterations = std::get<3>(input);

  while (iterations_done_ < max_iterations) {
    if (!ProcessIteration(input, epsilon, left_bound, right_bound)) {
      break;
    }
    ++iterations_done_;
  }

  return true;
}

bool StronginSearchMpi::PostProcessingImpl() {
  if (rank_ == 0) {
    GetOutput() = best_value_;
  }

  MPI_Bcast(&best_value_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank_ != 0) {
    GetOutput() = best_value_;
  }
  return true;
}

double StronginSearchMpi::ComputeGlobalSlope() const {
  const int intervals = static_cast<int>(points_.size()) - 1;
  double local_max_slope = 0.0;
  for (int i = 0; i < intervals; ++i) {
    const auto left_index = static_cast<std::size_t>(i);
    const auto &left = points_[left_index];
    const auto &right = points_[left_index + 1];
    const double delta = right.x - left.x;
    if (delta <= 0.0) {
      continue;
    }
    const double diff = right.value - left.value;
    const double slope = std::fabs(diff) / delta;
    local_max_slope = std::max(local_max_slope, slope);
  }

  double global_max_slope = 0.0;
  MPI_Allreduce(&local_max_slope, &global_max_slope, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return global_max_slope;
}

std::pair<int, int> StronginSearchMpi::IntervalRange(int intervals) const {
  const int base = intervals / world_size_;
  const int remainder = intervals % world_size_;
  const int start = (base * rank_) + std::min(rank_, remainder);
  const int count = base + (rank_ < remainder ? 1 : 0);
  return {start, start + count};
}

std::pair<double, int> StronginSearchMpi::EvaluateIntervals(int start, int end, double m) const {
  double local_best_value = -std::numeric_limits<double>::infinity();
  int local_best_index = -1;
  for (int idx = start; idx < end; ++idx) {
    const auto left_index = static_cast<std::size_t>(idx);
    const auto &left = points_[left_index];
    const auto &right = points_[left_index + 1];
    const double delta = right.x - left.x;
    if (delta <= 0.0) {
      continue;
    }
    const double diff = right.value - left.value;
    const double candidate = (m * delta) + ((diff * diff) / (m * delta)) - (2.0 * (right.value + left.value));
    if (candidate > local_best_value) {
      local_best_value = candidate;
      local_best_index = idx;
    }
  }

  IntervalCharacteristic local{.value = local_best_value, .index = local_best_index};
  IntervalCharacteristic global{};
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  return {global.value, global.index};
}

bool StronginSearchMpi::TryInsertPoint(const InType &input, int best_index, double epsilon, double m, double left_bound,
                                       double right_bound, int &insert_index, double &new_point, double &new_value) {
  const auto left_index = static_cast<std::size_t>(best_index);
  const auto &left = points_[left_index];
  const auto &right = points_[left_index + 1];
  const double interval_length = right.x - left.x;
  if (interval_length < epsilon) {
    return false;
  }

  new_point = (0.5 * (left.x + right.x)) - ((right.value - left.value) / (2.0 * m));
  new_point = std::clamp(new_point, left_bound, right_bound);
  const bool already_used = std::ranges::any_of(points_, [new_point](const SamplePoint &point) {
    return std::fabs(point.x - new_point) < std::numeric_limits<double>::epsilon();
  });
  if (already_used) {
    return false;
  }

  new_value = Evaluate(input, new_point);
  if (new_value < best_value_) {
    best_value_ = new_value;
    best_x_ = new_point;
  }
  auto insert_it = std::ranges::upper_bound(points_, SamplePoint{.x = new_point, .value = new_value}, Comparator);
  insert_index = static_cast<int>(insert_it - points_.begin());
  points_.insert(insert_it, SamplePoint{.x = new_point, .value = new_value});
  return true;
}

void StronginSearchMpi::BroadcastInsertionData(int &continue_flag, int &insert_index, double &new_point,
                                               double &new_value) {
  MPI_Bcast(&continue_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (continue_flag == 0) {
    return;
  }
  MPI_Bcast(&insert_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&new_point, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&new_value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool StronginSearchMpi::ProcessIteration(const InType &input, double epsilon, double left_bound, double right_bound) {
  if (points_.size() < 2) {
    return false;
  }
  const int intervals = static_cast<int>(points_.size()) - 1;
  const double global_max_slope = ComputeGlobalSlope();
  const double m = global_max_slope > 0.0 ? kReliability * global_max_slope : 1.0;
  const auto [start, end] = IntervalRange(intervals);
  const auto interval_selection = EvaluateIntervals(start, end, m);
  const int best_index = interval_selection.second;
  if (best_index < 0 || best_index >= intervals) {
    return false;
  }

  int insert_index = 0;
  double new_point = 0.0;
  double new_value = 0.0;
  int continue_flag = 0;
  if (rank_ == 0) {
    continue_flag =
        TryInsertPoint(input, best_index, epsilon, m, left_bound, right_bound, insert_index, new_point, new_value) ? 1
                                                                                                                   : 0;
  }

  BroadcastInsertionData(continue_flag, insert_index, new_point, new_value);
  if (continue_flag == 0) {
    return false;
  }

  if (rank_ != 0) {
    points_.insert(points_.begin() + insert_index, SamplePoint{.x = new_point, .value = new_value});
    if (new_value < best_value_) {
      best_value_ = new_value;
      best_x_ = new_point;
    }
  }

  MPI_Bcast(&best_value_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return true;
}

}  // namespace global_search_strongin
