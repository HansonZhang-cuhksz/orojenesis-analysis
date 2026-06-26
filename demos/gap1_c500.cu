#include <mcr/mc_runtime_api.h>
#include <mcpti/mcpti.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kElementBytes = 4;

struct GapInfo {
  std::string name;
  std::uint64_t total_elements = 0;
  std::uint64_t effectual_elements = 0;

  std::uint64_t total_bytes() const { return total_elements * kElementBytes; }
  std::uint64_t effectual_bytes() const {
    return effectual_elements * kElementBytes;
  }
  double gap1_percent() const {
    return 100.0 * static_cast<double>(effectual_elements) /
           static_cast<double>(total_elements);
  }
};

struct MeasuredTraffic {
  std::uint64_t dram_read_bytes = 0;
  std::uint64_t dram_write_bytes = 0;
  std::uint64_t l2_read_transaction_bytes = 0;
  std::uint64_t l2_write_transaction_bytes = 0;
};

bool check_runtime(mcError_t error, char const *call) {
  if (error == mcSuccess) {
    return true;
  }
  std::cerr << call << " failed: " << mcGetErrorString(error) << std::endl;
  return false;
}

#define MC_CHECK(call)       \
  do {                       \
    if (!check_runtime((call), #call)) { \
      return false;          \
    }                        \
  } while (0)

std::uint64_t metric_value_to_uint64(MCpti_MetricValueKind kind,
                                     MCpti_MetricValue const &value) {
  switch (kind) {
    case MCPTI_METRIC_VALUE_KIND_UINT64:
      return value.metricValueUint64;
    case MCPTI_METRIC_VALUE_KIND_INT64:
      return static_cast<std::uint64_t>(value.metricValueInt64);
    case MCPTI_METRIC_VALUE_KIND_DOUBLE:
      return static_cast<std::uint64_t>(value.metricValueDouble);
    case MCPTI_METRIC_VALUE_KIND_THROUGHPUT:
      return value.metricValueThroughput;
    default:
      return 0;
  }
}

bool set_profile_all_instances(MCpti_EventGroup group) {
  uint32_t profile_all = 1;
  MCptiResult result = mcptiEventGroupSetAttribute(
      group,
      MCPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
      sizeof(profile_all),
      &profile_all);
  if (result != MCPTI_SUCCESS) {
    std::cerr << "mcptiEventGroupSetAttribute failed with code " << result
              << std::endl;
    return false;
  }
  return true;
}

bool read_event_group(MCdevice device,
                      MCpti_EventGroup group,
                      std::map<MCpti_EventID, std::uint64_t> &event_values) {
  uint32_t num_events = 0;
  size_t attr_size = sizeof(num_events);
  MCptiResult result = mcptiEventGroupGetAttribute(
      group, MCPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &attr_size, &num_events);
  if (result != MCPTI_SUCCESS) {
    std::cerr << "mcptiEventGroupGetAttribute(NUM_EVENTS) failed with code "
              << result << std::endl;
    return false;
  }

  uint32_t instance_count = 1;
  attr_size = sizeof(instance_count);
  result = mcptiEventGroupGetAttribute(group,
                                       MCPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                       &attr_size,
                                       &instance_count);
  if (result != MCPTI_SUCCESS) {
    std::cerr << "mcptiEventGroupGetAttribute(INSTANCE_COUNT) failed with code "
              << result << std::endl;
    return false;
  }

  MCpti_EventDomainID domain = 0;
  attr_size = sizeof(domain);
  result = mcptiEventGroupGetAttribute(group,
                                       MCPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                                       &attr_size,
                                       &domain);
  if (result != MCPTI_SUCCESS) {
    std::cerr << "mcptiEventGroupGetAttribute(EVENT_DOMAIN_ID) failed with code "
              << result << std::endl;
    return false;
  }

  uint32_t total_instance_count = instance_count;
  attr_size = sizeof(total_instance_count);
  result = mcptiDeviceGetEventDomainAttribute(
      device,
      domain,
      MCPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
      &attr_size,
      &total_instance_count);
  if (result != MCPTI_SUCCESS) {
    total_instance_count = instance_count;
  }

  std::vector<MCpti_EventID> event_ids(num_events);
  attr_size = event_ids.size() * sizeof(MCpti_EventID);
  result = mcptiEventGroupGetAttribute(group,
                                       MCPTI_EVENT_GROUP_ATTR_EVENTS,
                                       &attr_size,
                                       event_ids.data());
  if (result != MCPTI_SUCCESS) {
    std::cerr << "mcptiEventGroupGetAttribute(EVENTS) failed with code "
              << result << std::endl;
    return false;
  }

  for (MCpti_EventID event_id : event_ids) {
    std::vector<std::uint64_t> values(instance_count, 0);
    size_t value_bytes = values.size() * sizeof(std::uint64_t);
    result = mcptiEventGroupReadEvent(group,
                                      MCPTI_EVENT_READ_FLAG_NONE,
                                      event_id,
                                      &value_bytes,
                                      values.data());
    if (result != MCPTI_SUCCESS) {
      std::cerr << "mcptiEventGroupReadEvent failed with code " << result
                << std::endl;
      return false;
    }

    std::uint64_t sum = 0;
    for (std::uint64_t value : values) {
      if (value != MCPTI_EVENT_OVERFLOW && value != MCPTI_EVENT_INVALID) {
        sum += value;
      }
    }
    if (instance_count != 0) {
      sum = (sum * total_instance_count) / instance_count;
    }
    event_values[event_id] = sum;
  }

  return true;
}

bool collect_metric(std::string const &metric_name,
                    MCdevice device,
                    MCcontext context,
                    std::function<bool()> const &launch,
                    std::uint64_t &metric_value) {
  MCpti_MetricID metric = 0;
  MCptiResult result =
      mcptiMetricGetIdFromName(device, metric_name.c_str(), &metric);
  if (result != MCPTI_SUCCESS) {
    std::cerr << "Metric not available: " << metric_name << " code=" << result
              << std::endl;
    return false;
  }

  MCpti_MetricValueKind value_kind = MCPTI_METRIC_VALUE_KIND_UINT64;
  size_t attr_size = sizeof(value_kind);
  result = mcptiMetricGetAttribute(metric,
                                   MCPTI_METRIC_ATTR_VALUE_KIND,
                                   &attr_size,
                                   &value_kind);
  if (result != MCPTI_SUCCESS) {
    std::cerr << "mcptiMetricGetAttribute(VALUE_KIND) failed with code "
              << result << std::endl;
    return false;
  }

  uint32_t num_metric_events = 0;
  result = mcptiMetricGetNumEvents(metric, &num_metric_events);
  if (result != MCPTI_SUCCESS || num_metric_events == 0) {
    std::cerr << "mcptiMetricGetNumEvents failed for " << metric_name
              << " code=" << result << std::endl;
    return false;
  }
  std::vector<MCpti_EventID> metric_events(num_metric_events);
  size_t event_bytes = metric_events.size() * sizeof(MCpti_EventID);
  result = mcptiMetricEnumEvents(metric, &event_bytes, metric_events.data());
  if (result != MCPTI_SUCCESS) {
    std::cerr << "mcptiMetricEnumEvents failed for " << metric_name
              << " code=" << result << std::endl;
    return false;
  }

  MCpti_EventGroupSets *group_sets = nullptr;
  result = mcptiMetricCreateEventGroupSets(
      context, sizeof(metric), &metric, &group_sets);
  if (result != MCPTI_SUCCESS || group_sets == nullptr) {
    std::cerr << "mcptiMetricCreateEventGroupSets failed for " << metric_name
              << " code=" << result << std::endl;
    return false;
  }

  std::map<MCpti_EventID, std::uint64_t> event_values;
  std::uint64_t elapsed_ns = 0;
  bool ok = true;

  result = mcptiSetEventCollectionMode(context, MCPTI_EVENT_COLLECTION_MODE_KERNEL);
  if (result != MCPTI_SUCCESS) {
    std::cerr << "mcptiSetEventCollectionMode failed with code " << result
              << std::endl;
    ok = false;
  }

  for (uint32_t set_idx = 0; ok && set_idx < group_sets->numSets; ++set_idx) {
    MCpti_EventGroupSet *set = &group_sets->sets[set_idx];
    for (uint32_t group_idx = 0; ok && group_idx < set->numEventGroups;
         ++group_idx) {
      ok = set_profile_all_instances(set->eventGroups[group_idx]);
    }

    result = mcptiEventGroupSetEnable(set);
    if (ok && result != MCPTI_SUCCESS) {
      std::cerr << "mcptiEventGroupSetEnable failed with code " << result
                << std::endl;
      ok = false;
    }

    auto start = std::chrono::steady_clock::now();
    if (ok) {
      ok = launch();
    }
    auto end = std::chrono::steady_clock::now();
    elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count());

    for (uint32_t group_idx = 0; ok && group_idx < set->numEventGroups;
         ++group_idx) {
      ok = read_event_group(device, set->eventGroups[group_idx], event_values);
    }

    result = mcptiEventGroupSetDisable(set);
    if (result != MCPTI_SUCCESS) {
      std::cerr << "mcptiEventGroupSetDisable failed with code " << result
                << std::endl;
      ok = false;
    }
  }

  if (ok) {
    std::vector<std::uint64_t> values(metric_events.size(), 0);
    for (std::size_t i = 0; i < metric_events.size(); ++i) {
      values[i] = event_values[metric_events[i]];
    }

    MCpti_MetricValue raw_value = {};
    result = mcptiMetricGetValue(device,
                                 metric,
                                 metric_events.size() * sizeof(MCpti_EventID),
                                 metric_events.data(),
                                 values.size() * sizeof(std::uint64_t),
                                 values.data(),
                                 elapsed_ns,
                                 &raw_value);
    if (result != MCPTI_SUCCESS) {
      std::cerr << "mcptiMetricGetValue failed for " << metric_name
                << " code=" << result << std::endl;
      ok = false;
    } else {
      metric_value = metric_value_to_uint64(value_kind, raw_value);
    }
  }

  mcptiEventGroupSetsDestroy(group_sets);
  return ok;
}

bool measure_traffic(std::function<bool()> const &launch,
                     MeasuredTraffic &traffic) {
  MC_CHECK(mcDeviceSynchronize());

  MCdevice device = 0;
  MC_CHECK(mcDeviceGet(&device, 0));
  MCcontext context = nullptr;
  MC_CHECK(mcCtxGetCurrent(&context));
  if (context == nullptr) {
    std::cerr << "No active MACA context for MCPTI profiling" << std::endl;
    return false;
  }

  std::uint64_t l2_read_transactions = 0;
  std::uint64_t l2_write_transactions = 0;
  if (!collect_metric("dram_read_bytes", device, context, launch,
                      traffic.dram_read_bytes) ||
      !collect_metric("dram_write_bytes", device, context, launch,
                      traffic.dram_write_bytes) ||
      !collect_metric("l2_read_transactions", device, context, launch,
                      l2_read_transactions) ||
      !collect_metric("l2_write_transactions", device, context, launch,
                      l2_write_transactions)) {
    return false;
  }

  traffic.l2_read_transaction_bytes = l2_read_transactions * 32;
  traffic.l2_write_transaction_bytes = l2_write_transactions * 32;
  return true;
}

std::vector<float> sequence(std::size_t count) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < values.size(); ++i) {
    int value = static_cast<int>(i % 17) - 8;
    values[i] = static_cast<float>(value) * 0.03125f;
  }
  return values;
}

bool copy_to_device(float *device, std::vector<float> const &host) {
  MC_CHECK(mcMemcpy(device,
                    host.data(),
                    host.size() * sizeof(float),
                    mcMemcpyHostToDevice));
  return true;
}

bool allocate(float **ptr, std::size_t count) {
  MC_CHECK(mcMalloc(reinterpret_cast<void **>(ptr), count * sizeof(float)));
  return true;
}

void print_result(GapInfo const &gap, MeasuredTraffic const &traffic) {
  std::uint64_t dram_total = traffic.dram_read_bytes + traffic.dram_write_bytes;
  std::uint64_t l2_total = traffic.l2_read_transaction_bytes +
                           traffic.l2_write_transaction_bytes;
  double dram_vs_total =
      static_cast<double>(dram_total) / static_cast<double>(gap.total_bytes());
  double l2_vs_total =
      static_cast<double>(l2_total) / static_cast<double>(gap.total_bytes());
  std::cout << gap.name << "\n"
            << "  Gap1 effectual buffer: " << gap.effectual_bytes()
            << " B\n"
            << "  Total operand size: " << gap.total_bytes() << " B\n"
            << "  Gap1 ratio: " << gap.gap1_percent() << "%\n"
            << "  DRAM read: " << traffic.dram_read_bytes << " B\n"
            << "  DRAM write: " << traffic.dram_write_bytes << " B\n"
            << "  DRAM total: " << dram_total << " B\n"
            << "  DRAM total / total operand size: " << dram_vs_total
            << "x\n"
            << "  L2 read transaction bytes: "
            << traffic.l2_read_transaction_bytes << " B\n"
            << "  L2 write transaction bytes: "
            << traffic.l2_write_transaction_bytes << " B\n"
            << "  L2 transaction total: " << l2_total << " B\n"
            << "  L2 transaction total / total operand size: " << l2_vs_total
            << "x\n";
}

}  // namespace

__global__ void vadd_kernel(float const *a, float const *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void gemv_kernel(float const *a,
                            float const *x,
                            float *y,
                            int m,
                            int k) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m) {
    float sum = 0.0f;
    for (int col = 0; col < k; ++col) {
      sum += a[row * k + col] * x[col];
    }
    y[row] = sum;
  }
}

__global__ void gemm_kernel(float const *a,
                            float const *b,
                            float *c,
                            int m,
                            int n,
                            int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    float sum = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
      sum += a[row * k + inner] * b[inner * n + col];
    }
    c[row * n + col] = sum;
  }
}

__global__ void conv2d_kernel(float const *input,
                              float const *weights,
                              float *output,
                              int p_size,
                              int q_size,
                              int channels,
                              int filters,
                              int r_size,
                              int s_size,
                              int input_p,
                              int input_q,
                              int stride,
                              int dilation) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = p_size * q_size * filters;
  if (idx >= total) {
    return;
  }

  int filter = idx % filters;
  int q = (idx / filters) % q_size;
  int p = idx / (filters * q_size);
  float sum = 0.0f;
  for (int c = 0; c < channels; ++c) {
    for (int r = 0; r < r_size; ++r) {
      for (int s = 0; s < s_size; ++s) {
        int in_p = (p * stride + r * dilation) % input_p;
        int in_q = (q * stride + s * dilation) % input_q;
        float x = input[(in_p * input_q + in_q) * channels + c];
        float w = weights[((c * filters + filter) * r_size + r) * s_size + s];
        sum += x * w;
      }
    }
  }
  output[idx] = sum;
}

bool run_vadd() {
  GapInfo gap{"1m vadd", 3'000'000, 3};
  int n = 1'000'000;
  auto a = sequence(n);
  auto b = sequence(n);
  float *da = nullptr;
  float *db = nullptr;
  float *dc = nullptr;
  if (!allocate(&da, n) || !allocate(&db, n) || !allocate(&dc, n) ||
      !copy_to_device(da, a) || !copy_to_device(db, b)) {
    return false;
  }

  auto launch = [&]() -> bool {
    int block = 256;
    int grid = (n + block - 1) / block;
    vadd_kernel<<<grid, block>>>(da, db, dc, n);
    MC_CHECK(mcDeviceSynchronize());
    return true;
  };

  MeasuredTraffic traffic;
  bool ok = measure_traffic(launch, traffic);
  print_result(gap, traffic);
  mcFree(da);
  mcFree(db);
  mcFree(dc);
  return ok;
}

bool run_gemv() {
  int m = 1000;
  int k = 1000;
  GapInfo gap{"1k x 1k GEMV",
              static_cast<std::uint64_t>(m) * k + k + m,
              static_cast<std::uint64_t>(k + m)};
  auto a = sequence(static_cast<std::size_t>(m) * k);
  auto x = sequence(k);
  float *da = nullptr;
  float *dx = nullptr;
  float *dy = nullptr;
  if (!allocate(&da, a.size()) || !allocate(&dx, x.size()) ||
      !allocate(&dy, m) || !copy_to_device(da, a) || !copy_to_device(dx, x)) {
    return false;
  }

  auto launch = [&]() -> bool {
    int block = 256;
    int grid = (m + block - 1) / block;
    gemv_kernel<<<grid, block>>>(da, dx, dy, m, k);
    MC_CHECK(mcDeviceSynchronize());
    return true;
  };

  MeasuredTraffic traffic;
  bool ok = measure_traffic(launch, traffic);
  print_result(gap, traffic);
  mcFree(da);
  mcFree(dx);
  mcFree(dy);
  return ok;
}

bool run_gemm() {
  int m = 1000;
  int n = 1000;
  int k = 1000;
  GapInfo gap{"1k x 1k x 1k GEMM",
              static_cast<std::uint64_t>(m) * k +
                  static_cast<std::uint64_t>(k) * n +
                  static_cast<std::uint64_t>(m) * n,
              static_cast<std::uint64_t>(m) * n};
  auto a = sequence(static_cast<std::size_t>(m) * k);
  auto b = sequence(static_cast<std::size_t>(k) * n);
  float *da = nullptr;
  float *db = nullptr;
  float *dc = nullptr;
  if (!allocate(&da, a.size()) || !allocate(&db, b.size()) ||
      !allocate(&dc, static_cast<std::size_t>(m) * n) ||
      !copy_to_device(da, a) || !copy_to_device(db, b)) {
    return false;
  }

  auto launch = [&]() -> bool {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    gemm_kernel<<<grid, block>>>(da, db, dc, m, n, k);
    MC_CHECK(mcDeviceSynchronize());
    return true;
  };

  MeasuredTraffic traffic;
  bool ok = measure_traffic(launch, traffic);
  print_result(gap, traffic);
  mcFree(da);
  mcFree(db);
  mcFree(dc);
  return ok;
}

bool run_conv(std::string const &name,
              int p,
              int q,
              int channels,
              int filters,
              int r,
              int s,
              int input_p,
              int input_q,
              int stride,
              int dilation) {
  std::uint64_t input_elements =
      static_cast<std::uint64_t>(input_p) * input_q * channels;
  std::uint64_t weight_elements =
      static_cast<std::uint64_t>(channels) * filters * r * s;
  std::uint64_t output_elements =
      static_cast<std::uint64_t>(p) * q * filters;
  GapInfo gap{name,
              input_elements + weight_elements + output_elements,
              std::min({input_elements, weight_elements, output_elements})};

  auto input = sequence(input_elements);
  auto weights = sequence(weight_elements);
  float *di = nullptr;
  float *dw = nullptr;
  float *do_ = nullptr;
  if (!allocate(&di, input.size()) || !allocate(&dw, weights.size()) ||
      !allocate(&do_, output_elements) || !copy_to_device(di, input) ||
      !copy_to_device(dw, weights)) {
    return false;
  }

  auto launch = [&]() -> bool {
    int total = static_cast<int>(output_elements);
    int block = 256;
    int grid = (total + block - 1) / block;
    conv2d_kernel<<<grid, block>>>(di,
                                   dw,
                                   do_,
                                   p,
                                   q,
                                   channels,
                                   filters,
                                   r,
                                   s,
                                   input_p,
                                   input_q,
                                   stride,
                                   dilation);
    MC_CHECK(mcDeviceSynchronize());
    return true;
  };

  MeasuredTraffic traffic;
  bool ok = measure_traffic(launch, traffic);
  print_result(gap, traffic);
  mcFree(di);
  mcFree(dw);
  mcFree(do_);
  return ok;
}

int main() {
  if (!check_runtime(mcInit(0), "mcInit") ||
      !check_runtime(mcSetDevice(0), "mcSetDevice")) {
    return EXIT_FAILURE;
  }

  bool ok = true;
  ok = run_vadd() && ok;
  ok = run_gemv() && ok;
  ok = run_gemm() && ok;
  ok = run_conv("1x1 conv", 16, 16, 64, 64, 1, 1, 16, 16, 1, 1) && ok;
  ok = run_conv("3x3 conv", 16, 16, 64, 64, 3, 3, 17, 17, 1, 1) && ok;
  ok = run_conv("3x3 conv stride 2", 16, 16, 64, 64, 3, 3, 32, 32, 2, 1) && ok;
  ok = run_conv("5x5 conv", 16, 16, 64, 64, 5, 5, 19, 19, 1, 1) && ok;

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
