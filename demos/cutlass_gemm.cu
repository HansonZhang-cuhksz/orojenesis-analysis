#if defined(USE_MCTLASS)
#include <mcr/mc_runtime_api.h>
#include <mcpti/mcpti.h>

#include <mctlass/frontend_op/gemm_group_config.h>
#include <mctlass/gemm/gemm.h>
#include <mctlass/gemm/threadblock/threadblock_swizzle.h>
#include <mctlass/half.h>
#include <mctlass/layout/matrix.h>
#include <mctlass/mctlass.h>
#else
#include <cuda_runtime.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/status.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(USE_MCTLASS)
namespace gemm_backend = mctlass;
using RuntimeError = mcError_t;
#define RUNTIME_MALLOC mcMalloc
#define RUNTIME_FREE mcFree
#define RUNTIME_MEMCPY mcMemcpy
#define RUNTIME_DEVICE_SYNCHRONIZE mcDeviceSynchronize
#define RUNTIME_GET_ERROR_STRING mcGetErrorString
#define RUNTIME_MEMCPY_HOST_TO_DEVICE mcMemcpyHostToDevice
#define RUNTIME_MEMCPY_DEVICE_TO_HOST mcMemcpyDeviceToHost
#define GEMM_GET_STATUS_STRING mctlassGetStatusString
#define GEMM_LIBRARY_NAME "MCTLASS"
#else
namespace gemm_backend = cutlass;
using RuntimeError = cudaError_t;
#define RUNTIME_MALLOC cudaMalloc
#define RUNTIME_FREE cudaFree
#define RUNTIME_MEMCPY cudaMemcpy
#define RUNTIME_DEVICE_SYNCHRONIZE cudaDeviceSynchronize
#define RUNTIME_GET_ERROR_STRING cudaGetErrorString
#define RUNTIME_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define RUNTIME_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define GEMM_GET_STATUS_STRING cutlassGetStatusString
#define GEMM_LIBRARY_NAME "CUTLASS"
#endif

using Element = float;

#if defined(USE_MCTLASS)
constexpr int kTrafficTileM = 128;
constexpr int kTrafficTileN = 128;
constexpr int kTrafficTileK = 32;
#else
constexpr int kTrafficTileM = 128;
constexpr int kTrafficTileN = 128;
constexpr int kTrafficTileK = 8;
#endif

#define RUNTIME_CHECK(call)                                                   \
  do {                                                                        \
    if (!check_runtime((call), __FILE__, __LINE__)) {                         \
      return EXIT_FAILURE;                                                     \
    }                                                                         \
  } while (0)

namespace {

bool check_runtime(RuntimeError error, char const *file, int line) {
  int error_code = static_cast<int>(error);
  if (error_code == 0) {
    return true;
  }

  std::cerr << "Runtime error at " << file << ":" << line
            << ": code=" << error_code
            << " message=" << RUNTIME_GET_ERROR_STRING(error) << std::endl;
  return false;
}

struct TrafficReport {
  std::uint64_t algorithmic_minimum_bytes = 0;
  std::uint64_t hbm_read_bytes = 0;
  std::uint64_t hbm_write_bytes = 0;
  std::uint64_t l1_l2_read_bytes = 0;
  std::uint64_t l1_l2_write_bytes = 0;
};

std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

std::uint64_t tile_extent(std::uint64_t size,
                          std::uint64_t tile,
                          std::uint64_t index) {
  std::uint64_t start = index * tile;
  return std::min(tile, size - start);
}

TrafficReport estimate_traffic_bytes(int m, int n, int k, bool reads_c) {
  std::uint64_t m_size = static_cast<std::uint64_t>(m);
  std::uint64_t n_size = static_cast<std::uint64_t>(n);
  std::uint64_t k_size = static_cast<std::uint64_t>(k);
  std::uint64_t element_bytes = sizeof(Element);

  std::uint64_t a_bytes = m_size * k_size * element_bytes;
  std::uint64_t b_bytes = k_size * n_size * element_bytes;
  std::uint64_t c_or_d_bytes = m_size * n_size * element_bytes;

  TrafficReport report;
  report.algorithmic_minimum_bytes =
      a_bytes + b_bytes + c_or_d_bytes + (reads_c ? c_or_d_bytes : 0);
  report.hbm_read_bytes = a_bytes + b_bytes + (reads_c ? c_or_d_bytes : 0);
  report.hbm_write_bytes = c_or_d_bytes;

  std::uint64_t tile_m = static_cast<std::uint64_t>(kTrafficTileM);
  std::uint64_t tile_n = static_cast<std::uint64_t>(kTrafficTileN);
  std::uint64_t tile_k = static_cast<std::uint64_t>(kTrafficTileK);
  std::uint64_t m_tiles = ceil_div(m_size, tile_m);
  std::uint64_t n_tiles = ceil_div(n_size, tile_n);
  std::uint64_t k_tiles = ceil_div(k_size, tile_k);

  for (std::uint64_t mi = 0; mi < m_tiles; ++mi) {
    std::uint64_t m_count = tile_extent(m_size, tile_m, mi);
    for (std::uint64_t ni = 0; ni < n_tiles; ++ni) {
      std::uint64_t n_count = tile_extent(n_size, tile_n, ni);

      if (reads_c) {
        report.l1_l2_read_bytes += m_count * n_count * element_bytes;
      }
      report.l1_l2_write_bytes += m_count * n_count * element_bytes;

      for (std::uint64_t ki = 0; ki < k_tiles; ++ki) {
        std::uint64_t k_count = tile_extent(k_size, tile_k, ki);
        report.l1_l2_read_bytes +=
            (m_count * k_count + k_count * n_count) * element_bytes;
      }
    }
  }

  return report;
}

void print_traffic_report(TrafficReport const &report) {
  std::uint64_t hbm_total = report.hbm_read_bytes + report.hbm_write_bytes;
  std::uint64_t l1_l2_total =
      report.l1_l2_read_bytes + report.l1_l2_write_bytes;
  double hbm_ratio = static_cast<double>(hbm_total) /
                     static_cast<double>(report.algorithmic_minimum_bytes);
  double l1_l2_ratio = static_cast<double>(l1_l2_total) /
                       static_cast<double>(report.algorithmic_minimum_bytes);

  std::cout << "Traffic estimate (Bytes):\n"
            << "  Algorithmic minimum: "
            << report.algorithmic_minimum_bytes << "\n"
            << "  HBM read: " << report.hbm_read_bytes << "\n"
            << "  HBM write: " << report.hbm_write_bytes << "\n"
            << "  HBM total: " << hbm_total << "\n"
            << "  HBM / algorithmic minimum: " << hbm_ratio << "x\n"
            << "  L1-L2 read: " << report.l1_l2_read_bytes << "\n"
            << "  L1-L2 write: " << report.l1_l2_write_bytes << "\n"
            << "  L1-L2 total: " << l1_l2_total << "\n"
            << "  L1-L2 / algorithmic minimum: " << l1_l2_ratio << "x"
            << std::endl;
}

#if defined(USE_MCTLASS)
std::string mcpti_metric_name(MCpti_MetricID metric) {
  char buffer[256] = {};
  size_t size = sizeof(buffer);
  if (mcptiMetricGetAttribute(metric,
                              MCPTI_METRIC_ATTR_NAME,
                              &size,
                              buffer) != MCPTI_SUCCESS) {
    return {};
  }
  return std::string(buffer, strnlen(buffer, sizeof(buffer)));
}

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
      if (value == MCPTI_EVENT_OVERFLOW || value == MCPTI_EVENT_INVALID) {
        continue;
      }
      sum += value;
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
    if (!ok) {
      break;
    }

    result = mcptiEventGroupSetEnable(set);
    if (result != MCPTI_SUCCESS) {
      std::cerr << "mcptiEventGroupSetEnable failed with code " << result
                << std::endl;
      ok = false;
      break;
    }

    auto start = std::chrono::steady_clock::now();
    ok = launch();
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
      break;
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

void print_measured_traffic(TrafficReport const &minimum,
                            std::uint64_t dram_read,
                            std::uint64_t dram_write,
                            std::uint64_t l2_read_bytes,
                            std::uint64_t l2_write_bytes) {
  std::uint64_t dram_total = dram_read + dram_write;
  std::uint64_t l2_total = l2_read_bytes + l2_write_bytes;
  double dram_ratio = static_cast<double>(dram_total) /
                      static_cast<double>(minimum.algorithmic_minimum_bytes);
  double l2_ratio = static_cast<double>(l2_total) /
                    static_cast<double>(minimum.algorithmic_minimum_bytes);

  std::cout << "MCPTI measured traffic proxy (Bytes):\n"
            << "  Algorithmic minimum: "
            << minimum.algorithmic_minimum_bytes << "\n"
            << "  DRAM read: " << dram_read << "\n"
            << "  DRAM write: " << dram_write << "\n"
            << "  DRAM total: " << dram_total << "\n"
            << "  DRAM / algorithmic minimum: " << dram_ratio << "x\n"
            << "  L2 read transaction bytes: " << l2_read_bytes << "\n"
            << "  L2 write transaction bytes: " << l2_write_bytes << "\n"
            << "  L2 transaction total: " << l2_total << "\n"
            << "  L2 transaction / algorithmic minimum: " << l2_ratio << "x"
            << std::endl;
}
#endif

int parse_positive_int(char const *arg, char const *name) {
  char *end = nullptr;
  long value = std::strtol(arg, &end, 10);
  if (*arg == '\0' || *end != '\0' || value <= 0 ||
      value > static_cast<long>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(name) + " must be a positive integer");
  }
  return static_cast<int>(value);
}

float to_float(float value) {
  return value;
}

#if defined(USE_MCTLASS)
float to_float(mctlass::half_t value) {
  return static_cast<float>(value);
}
#endif

Element from_float(float value) {
  return Element(value);
}

void fill_random(std::vector<Element> &matrix) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (Element &value : matrix) {
    value = from_float(dist(rng));
  }
}

void reference_gemm(int m,
                    int n,
                    int k,
                    float alpha,
                    std::vector<Element> const &a,
                    std::vector<Element> const &b,
                    float beta,
                    std::vector<Element> const &c,
                    std::vector<float> &d) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float accum = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        accum += to_float(a[row * k + inner]) * to_float(b[inner * n + col]);
      }
      d[row * n + col] = alpha * accum + beta * to_float(c[row * n + col]);
    }
  }
}

float reference_element(int row,
                        int col,
                        int n,
                        int k,
                        float alpha,
                        std::vector<Element> const &a,
                        std::vector<Element> const &b,
                        float beta,
                        std::vector<Element> const &c) {
  float accum = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    accum += to_float(a[row * k + inner]) * to_float(b[inner * n + col]);
  }
  return alpha * accum + beta * to_float(c[row * n + col]);
}

bool almost_equal(std::vector<Element> const &actual,
                  std::vector<float> const &expected,
                  float tolerance,
                  float &max_abs_error,
                  std::size_t &max_error_index) {
  max_abs_error = 0.0f;
  max_error_index = 0;

  for (std::size_t i = 0; i < actual.size(); ++i) {
    float error = std::fabs(to_float(actual[i]) - expected[i]);
    if (error > max_abs_error) {
      max_abs_error = error;
      max_error_index = i;
    }
  }

  return max_abs_error <= tolerance;
}

bool almost_equal_sampled(int m,
                          int n,
                          int k,
                          float alpha,
                          std::vector<Element> const &a,
                          std::vector<Element> const &b,
                          float beta,
                          std::vector<Element> const &c,
                          std::vector<Element> const &actual,
                          float tolerance,
                          float &max_abs_error,
                          std::size_t &max_error_index) {
  std::uint64_t output_count = static_cast<std::uint64_t>(m) * n;
  std::uint64_t sample_count = std::min<std::uint64_t>(output_count, 1024);
  std::uint64_t stride = std::max<std::uint64_t>(1, output_count / sample_count);

  max_abs_error = 0.0f;
  max_error_index = 0;

  for (std::uint64_t sample = 0; sample < sample_count; ++sample) {
    std::uint64_t index = std::min(output_count - 1, sample * stride);
    int row = static_cast<int>(index / n);
    int col = static_cast<int>(index % n);
    float expected = reference_element(row, col, n, k, alpha, a, b, beta, c);
    float error = std::fabs(to_float(actual[index]) - expected);
    if (error > max_abs_error) {
      max_abs_error = error;
      max_error_index = static_cast<std::size_t>(index);
    }
  }

  return max_abs_error <= tolerance;
}

void print_usage(char const *program) {
  std::cerr << "Usage: " << program << " [--mcpti] [M N K]\n"
            << "Runs D = alpha * A * B + beta * C using " << GEMM_LIBRARY_NAME
            << " GEMM.\n"
            << "Defaults: M=N=K=256\n"
            << "MCTLASS FP32 requires K to be a multiple of 4.\n";
}

}  // namespace

int main(int argc, char **argv) {
  int m = 256;
  int n = 256;
  int k = 256;
  bool profile_metrics = false;

  int arg_index = 1;
  if (argc > arg_index && std::strcmp(argv[arg_index], "--mcpti") == 0) {
    profile_metrics = true;
    ++arg_index;
  }

  int remaining_args = argc - arg_index;
  if (remaining_args != 0 && remaining_args != 3) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  try {
    if (remaining_args == 3) {
      m = parse_positive_int(argv[arg_index], "M");
      n = parse_positive_int(argv[arg_index + 1], "N");
      k = parse_positive_int(argv[arg_index + 2], "K");
    }
  } catch (std::exception const &error) {
    std::cerr << error.what() << std::endl;
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

#if defined(USE_MCTLASS)
  if (k % 4 != 0) {
    std::cerr << "MCTLASS FP32 GEMM requires K to be a multiple of 4"
              << std::endl;
    return EXIT_FAILURE;
  }
#endif

#if !defined(USE_MCTLASS)
  if (profile_metrics) {
    std::cerr << "--mcpti is only supported for BACKEND=maca" << std::endl;
    return EXIT_FAILURE;
  }
#endif

  float alpha = 1.0f;
#if defined(USE_MCTLASS)
  float beta = 0.0f;
#else
  float beta = 1.0f;
#endif

  std::vector<Element> host_a(static_cast<std::size_t>(m) * k);
  std::vector<Element> host_b(static_cast<std::size_t>(k) * n);
  std::vector<Element> host_c(static_cast<std::size_t>(m) * n);
  std::vector<Element> host_d(static_cast<std::size_t>(m) * n, from_float(0.0f));
  std::vector<float> host_reference(static_cast<std::size_t>(m) * n, 0.0f);

  fill_random(host_a);
  fill_random(host_b);
  fill_random(host_c);

  std::vector<Element> host_b_device(host_b.size());
#if defined(USE_MCTLASS)
  for (int row = 0; row < k; ++row) {
    for (int col = 0; col < n; ++col) {
      host_b_device[static_cast<std::size_t>(col) * k + row] =
          host_b[static_cast<std::size_t>(row) * n + col];
    }
  }
#else
  host_b_device = host_b;
#endif

  Element *device_a = nullptr;
  Element *device_b = nullptr;
  Element *device_c = nullptr;
  Element *device_d = nullptr;

  RUNTIME_CHECK(RUNTIME_MALLOC(reinterpret_cast<void **>(&device_a),
                               host_a.size() * sizeof(Element)));
  RUNTIME_CHECK(RUNTIME_MALLOC(reinterpret_cast<void **>(&device_b),
                               host_b.size() * sizeof(Element)));
  RUNTIME_CHECK(RUNTIME_MALLOC(reinterpret_cast<void **>(&device_c),
                               host_c.size() * sizeof(Element)));
  RUNTIME_CHECK(RUNTIME_MALLOC(reinterpret_cast<void **>(&device_d),
                               host_d.size() * sizeof(Element)));

  RUNTIME_CHECK(RUNTIME_MEMCPY(device_a,
                               host_a.data(),
                               host_a.size() * sizeof(Element),
                               RUNTIME_MEMCPY_HOST_TO_DEVICE));
  RUNTIME_CHECK(RUNTIME_MEMCPY(device_b,
                               host_b_device.data(),
                               host_b.size() * sizeof(Element),
                               RUNTIME_MEMCPY_HOST_TO_DEVICE));
  RUNTIME_CHECK(RUNTIME_MEMCPY(device_c,
                               host_c.data(),
                               host_c.size() * sizeof(Element),
                               RUNTIME_MEMCPY_HOST_TO_DEVICE));

#if defined(USE_MCTLASS)
  using LayoutA = gemm_backend::layout::RowMajor;
  using LayoutB = gemm_backend::layout::ColumnMajor;
  using LayoutC = gemm_backend::layout::RowMajor;
#else
  using LayoutA = gemm_backend::layout::RowMajor;
  using LayoutB = gemm_backend::layout::RowMajor;
  using LayoutC = gemm_backend::layout::RowMajor;
#endif
  int lda = k;
#if defined(USE_MCTLASS)
  int ldb = k;
#else
  int ldb = n;
#endif
  int ldc = n;
  int ldd = n;

#if defined(USE_MCTLASS)
  mctlass::gemm::GemmCoord host_problem(m, n, k);
  mctlass::gemm::GemmCoord *device_problem = nullptr;
  Element **device_ptr_a = nullptr;
  Element **device_ptr_b = nullptr;
  Element **device_ptr_c = nullptr;
  Element **device_ptr_d = nullptr;
  int64_t *device_lda = nullptr;
  int64_t *device_ldb = nullptr;
  int64_t *device_ldc = nullptr;
  int64_t *device_ldd = nullptr;

  Element *host_ptr_a[] = {device_a};
  Element *host_ptr_b[] = {device_b};
  Element *host_ptr_c[] = {device_c};
  Element *host_ptr_d[] = {device_d};
  int64_t host_lda[] = {lda};
  int64_t host_ldb[] = {ldb};
  int64_t host_ldc[] = {ldc};
  int64_t host_ldd[] = {ldd};

  RUNTIME_CHECK(RUNTIME_MALLOC(reinterpret_cast<void **>(&device_problem),
                               sizeof(host_problem)));
  RUNTIME_CHECK(RUNTIME_MALLOC(reinterpret_cast<void **>(&device_ptr_a),
                               sizeof(host_ptr_a)));
  RUNTIME_CHECK(RUNTIME_MALLOC(reinterpret_cast<void **>(&device_ptr_b),
                               sizeof(host_ptr_b)));
  RUNTIME_CHECK(RUNTIME_MALLOC(reinterpret_cast<void **>(&device_ptr_c),
                               sizeof(host_ptr_c)));
  RUNTIME_CHECK(RUNTIME_MALLOC(reinterpret_cast<void **>(&device_ptr_d),
                               sizeof(host_ptr_d)));
  RUNTIME_CHECK(RUNTIME_MALLOC(reinterpret_cast<void **>(&device_lda),
                               sizeof(host_lda)));
  RUNTIME_CHECK(RUNTIME_MALLOC(reinterpret_cast<void **>(&device_ldb),
                               sizeof(host_ldb)));
  RUNTIME_CHECK(RUNTIME_MALLOC(reinterpret_cast<void **>(&device_ldc),
                               sizeof(host_ldc)));
  RUNTIME_CHECK(RUNTIME_MALLOC(reinterpret_cast<void **>(&device_ldd),
                               sizeof(host_ldd)));

  RUNTIME_CHECK(RUNTIME_MEMCPY(device_problem,
                               &host_problem,
                               sizeof(host_problem),
                               RUNTIME_MEMCPY_HOST_TO_DEVICE));
  RUNTIME_CHECK(RUNTIME_MEMCPY(device_ptr_a,
                               host_ptr_a,
                               sizeof(host_ptr_a),
                               RUNTIME_MEMCPY_HOST_TO_DEVICE));
  RUNTIME_CHECK(RUNTIME_MEMCPY(device_ptr_b,
                               host_ptr_b,
                               sizeof(host_ptr_b),
                               RUNTIME_MEMCPY_HOST_TO_DEVICE));
  RUNTIME_CHECK(RUNTIME_MEMCPY(device_ptr_c,
                               host_ptr_c,
                               sizeof(host_ptr_c),
                               RUNTIME_MEMCPY_HOST_TO_DEVICE));
  RUNTIME_CHECK(RUNTIME_MEMCPY(device_ptr_d,
                               host_ptr_d,
                               sizeof(host_ptr_d),
                               RUNTIME_MEMCPY_HOST_TO_DEVICE));
  RUNTIME_CHECK(RUNTIME_MEMCPY(device_lda,
                               host_lda,
                               sizeof(host_lda),
                               RUNTIME_MEMCPY_HOST_TO_DEVICE));
  RUNTIME_CHECK(RUNTIME_MEMCPY(device_ldb,
                               host_ldb,
                               sizeof(host_ldb),
                               RUNTIME_MEMCPY_HOST_TO_DEVICE));
  RUNTIME_CHECK(RUNTIME_MEMCPY(device_ldc,
                               host_ldc,
                               sizeof(host_ldc),
                               RUNTIME_MEMCPY_HOST_TO_DEVICE));
  RUNTIME_CHECK(RUNTIME_MEMCPY(device_ldd,
                               host_ldd,
                               sizeof(host_ldd),
                               RUNTIME_MEMCPY_HOST_TO_DEVICE));

  using ShapeMMAThreadBlock = mctlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = mctlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAOp = mctlass::gemm::GemmShape<16, 16, 4>;
  using DeviceGemm = typename GemmGroupConfig<
      Element,
      LayoutA,
      Element,
      LayoutB,
      Element,
      LayoutC,
      float,
      mctlass::arch::Sm80,
      ShapeMMAThreadBlock,
      ShapeMMAWarp,
      ShapeMMAOp,
      4,
      4,
      4,
      128,
      mctlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      Activation::None>::GemmGroup;
  DeviceGemm gemm;
  typename DeviceGemm::EpilogueOutputOp::Params epilogue_op(alpha, beta);
  typename DeviceGemm::Arguments args(device_problem,
                                      1,
                                      1,
                                      epilogue_op,
                                      device_ptr_a,
                                      device_ptr_b,
                                      device_ptr_d,
                                      device_ptr_d,
                                      device_lda,
                                      device_ldb,
                                      device_ldc,
                                      device_ldd,
                                      &host_problem);

  gemm_backend::Status status = gemm(args, nullptr);
#else
  using DeviceGemm = gemm_backend::gemm::device::Gemm<Element,
                                                      LayoutA,
                                                      Element,
                                                      LayoutB,
                                                      Element,
                                                      LayoutC>;

  DeviceGemm gemm;
  typename DeviceGemm::Arguments args({m, n, k},
                                      {device_a, lda},
                                      {device_b, ldb},
                                      {device_c, ldc},
                                      {device_d, ldd},
                                      {alpha, beta});

  gemm_backend::Status status = gemm.can_implement(args);
  if (status != gemm_backend::Status::kSuccess) {
    std::cerr << GEMM_LIBRARY_NAME << " cannot implement this GEMM: "
              << GEMM_GET_STATUS_STRING(status) << std::endl;
    return EXIT_FAILURE;
  }

  status = gemm(args);
#endif
  if (status != gemm_backend::Status::kSuccess) {
    std::cerr << GEMM_LIBRARY_NAME << " GEMM failed: "
              << GEMM_GET_STATUS_STRING(status) << std::endl;
    return EXIT_FAILURE;
  }

#if defined(USE_MCTLASS)
  std::uint64_t measured_dram_read = 0;
  std::uint64_t measured_dram_write = 0;
  std::uint64_t measured_l2_read_transactions = 0;
  std::uint64_t measured_l2_write_transactions = 0;
  if (profile_metrics) {
    RUNTIME_CHECK(RUNTIME_DEVICE_SYNCHRONIZE());

    MCdevice profile_device = 0;
    RUNTIME_CHECK(mcDeviceGet(&profile_device, 0));
    MCcontext profile_context = nullptr;
    RUNTIME_CHECK(mcCtxGetCurrent(&profile_context));
    if (profile_context == nullptr) {
      std::cerr << "No active MACA context for MCPTI profiling" << std::endl;
      return EXIT_FAILURE;
    }

    auto launch_profiled_gemm = [&]() -> bool {
      gemm_backend::Status profile_status = gemm(args, nullptr);
      if (profile_status != gemm_backend::Status::kSuccess) {
        std::cerr << GEMM_LIBRARY_NAME << " profiled GEMM failed: "
                  << GEMM_GET_STATUS_STRING(profile_status) << std::endl;
        return false;
      }
      return check_runtime(RUNTIME_DEVICE_SYNCHRONIZE(), __FILE__, __LINE__);
    };

    if (!collect_metric("dram_read_bytes",
                        profile_device,
                        profile_context,
                        launch_profiled_gemm,
                        measured_dram_read) ||
        !collect_metric("dram_write_bytes",
                        profile_device,
                        profile_context,
                        launch_profiled_gemm,
                        measured_dram_write) ||
        !collect_metric("l2_read_transactions",
                        profile_device,
                        profile_context,
                        launch_profiled_gemm,
                        measured_l2_read_transactions) ||
        !collect_metric("l2_write_transactions",
                        profile_device,
                        profile_context,
                        launch_profiled_gemm,
                        measured_l2_write_transactions)) {
      std::cerr << "MCPTI metric collection failed" << std::endl;
      return EXIT_FAILURE;
    }
  }
#endif

  RUNTIME_CHECK(RUNTIME_DEVICE_SYNCHRONIZE());
  RUNTIME_CHECK(RUNTIME_MEMCPY(host_d.data(),
                               device_d,
                               host_d.size() * sizeof(Element),
                               RUNTIME_MEMCPY_DEVICE_TO_HOST));

  float max_abs_error = 0.0f;
  std::size_t max_error_index = 0;
#if defined(USE_MCTLASS)
  float tolerance = std::max(1.0e-2f, 1.0e-4f * static_cast<float>(k));
#else
  float tolerance = std::max(1.0e-3f, 1.0e-5f * static_cast<float>(k));
#endif
  std::uint64_t reference_ops =
      static_cast<std::uint64_t>(m) * n * static_cast<std::uint64_t>(k);
  constexpr std::uint64_t kFullReferenceOpsLimit = 100000000;
  bool used_sampled_reference = reference_ops > kFullReferenceOpsLimit;
  bool passed = false;
  if (used_sampled_reference) {
    passed = almost_equal_sampled(m,
                                  n,
                                  k,
                                  alpha,
                                  host_a,
                                  host_b,
                                  beta,
                                  host_c,
                                  host_d,
                                  tolerance,
                                  max_abs_error,
                                  max_error_index);
  } else {
    reference_gemm(m, n, k, alpha, host_a, host_b, beta, host_c, host_reference);
    passed = almost_equal(host_d,
                          host_reference,
                          tolerance,
                          max_abs_error,
                          max_error_index);
  }

  RUNTIME_CHECK(RUNTIME_FREE(device_a));
  RUNTIME_CHECK(RUNTIME_FREE(device_b));
  RUNTIME_CHECK(RUNTIME_FREE(device_c));
  RUNTIME_CHECK(RUNTIME_FREE(device_d));
#if defined(USE_MCTLASS)
  RUNTIME_CHECK(RUNTIME_FREE(device_problem));
  RUNTIME_CHECK(RUNTIME_FREE(device_ptr_a));
  RUNTIME_CHECK(RUNTIME_FREE(device_ptr_b));
  RUNTIME_CHECK(RUNTIME_FREE(device_ptr_c));
  RUNTIME_CHECK(RUNTIME_FREE(device_ptr_d));
  RUNTIME_CHECK(RUNTIME_FREE(device_lda));
  RUNTIME_CHECK(RUNTIME_FREE(device_ldb));
  RUNTIME_CHECK(RUNTIME_FREE(device_ldc));
  RUNTIME_CHECK(RUNTIME_FREE(device_ldd));
#endif

  if (!passed) {
    std::cerr << "Verification failed for " << m << "x" << n << "x" << k
              << " GEMM. max_abs_error=" << max_abs_error
              << " at index " << max_error_index
              << ", tolerance=" << tolerance << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << GEMM_LIBRARY_NAME << " GEMM passed for M=" << m << " N=" << n
            << " K=" << k << " with max_abs_error=" << max_abs_error
            << " using "
            << (used_sampled_reference ? "sampled" : "full")
            << " verification" << std::endl;
  TrafficReport estimated_traffic =
      estimate_traffic_bytes(m, n, k, beta != 0.0f);
  print_traffic_report(estimated_traffic);
#if defined(USE_MCTLASS)
  if (profile_metrics) {
    print_measured_traffic(estimated_traffic,
                           measured_dram_read,
                           measured_dram_write,
                           measured_l2_read_transactions * 32,
                           measured_l2_write_transactions * 32);
  }
#endif

  return EXIT_SUCCESS;
}
