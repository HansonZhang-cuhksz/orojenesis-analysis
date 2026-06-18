#include <mcpti/mcpti.h>
#include <mcr/mc_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool contains_filter(std::string const &text, std::string const &filter) {
  return filter.empty() || lower(text).find(filter) != std::string::npos;
}

std::string metric_string(MCpti_MetricID metric, MCpti_MetricAttribute attr) {
  char buffer[4096] = {};
  size_t size = sizeof(buffer);
  MCptiResult result = mcptiMetricGetAttribute(metric, attr, &size, buffer);
  if (result != MCPTI_SUCCESS) {
    return {};
  }
  return std::string(buffer, strnlen(buffer, sizeof(buffer)));
}

std::string event_string(MCpti_EventID event, MCpti_EventAttribute attr);

std::string metric_value_kind(MCpti_MetricID metric) {
  MCpti_MetricValueKind kind = MCPTI_METRIC_VALUE_KIND_UINT64;
  size_t size = sizeof(kind);
  MCptiResult result =
      mcptiMetricGetAttribute(metric, MCPTI_METRIC_ATTR_VALUE_KIND, &size, &kind);
  if (result != MCPTI_SUCCESS) {
    return "unknown";
  }
  switch (kind) {
    case MCPTI_METRIC_VALUE_KIND_DOUBLE:
      return "double";
    case MCPTI_METRIC_VALUE_KIND_UINT64:
      return "uint64";
    case MCPTI_METRIC_VALUE_KIND_PERCENT:
      return "percent";
    case MCPTI_METRIC_VALUE_KIND_THROUGHPUT:
      return "throughput";
    case MCPTI_METRIC_VALUE_KIND_INT64:
      return "int64";
    case MCPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
      return "utilization";
    default:
      return "unknown";
  }
}

void print_metric_events(MCpti_MetricID metric) {
  uint32_t num_events = 0;
  if (mcptiMetricGetNumEvents(metric, &num_events) != MCPTI_SUCCESS ||
      num_events == 0) {
    return;
  }
  std::vector<MCpti_EventID> events(num_events);
  size_t bytes = events.size() * sizeof(MCpti_EventID);
  if (mcptiMetricEnumEvents(metric, &bytes, events.data()) != MCPTI_SUCCESS) {
    return;
  }
  std::cout << "    events:";
  for (MCpti_EventID event : events) {
    std::cout << " " << event_string(event, MCPTI_EVENT_ATTR_NAME);
  }
  std::cout << "\n";
}

std::string event_string(MCpti_EventID event, MCpti_EventAttribute attr) {
  char buffer[4096] = {};
  size_t size = sizeof(buffer);
  MCptiResult result = mcptiEventGetAttribute(event, attr, &size, buffer);
  if (result != MCPTI_SUCCESS) {
    return {};
  }
  return std::string(buffer, strnlen(buffer, sizeof(buffer)));
}

void check(MCptiResult result, char const *call) {
  if (result != MCPTI_SUCCESS) {
    std::cerr << call << " failed with MCPTI code " << result << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void check(mcError_t result, char const *call) {
  if (result != mcSuccess) {
    std::cerr << call << " failed with MACA code " << static_cast<int>(result)
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::string filter;
  if (argc > 2) {
    std::cerr << "Usage: " << argv[0] << " [substring-filter]\n";
    return EXIT_FAILURE;
  }
  if (argc == 2) {
    filter = lower(argv[1]);
  }

  check(mcInit(0), "mcInit");
  check(mcSetDevice(0), "mcSetDevice");

  MCdevice device = 0;
  check(mcDeviceGet(&device, 0), "mcDeviceGet");

  char device_name[256] = {};
  check(mcDeviceGetName(device_name, sizeof(device_name), device), "mcDeviceGetName");
  std::cout << "Device: " << device_name << "\n";

  uint32_t num_metrics = 0;
  check(mcptiDeviceGetNumMetrics(device, &num_metrics),
        "mcptiDeviceGetNumMetrics");
  std::vector<MCpti_MetricID> metrics(num_metrics);
  size_t metric_bytes = metrics.size() * sizeof(MCpti_MetricID);
  check(mcptiDeviceEnumMetrics(device, &metric_bytes, metrics.data()),
        "mcptiDeviceEnumMetrics");

  std::cout << "Metrics:\n";
  for (MCpti_MetricID metric : metrics) {
    std::string name = metric_string(metric, MCPTI_METRIC_ATTR_NAME);
    std::string short_desc =
        metric_string(metric, MCPTI_METRIC_ATTR_SHORT_DESCRIPTION);
    std::string long_desc =
        metric_string(metric, MCPTI_METRIC_ATTR_LONG_DESCRIPTION);
    std::string haystack = name + " " + short_desc + " " + long_desc;
    if (!contains_filter(haystack, filter)) {
      continue;
    }
    std::cout << "  " << name << " [" << metric_value_kind(metric) << "]\n";
    if (!short_desc.empty()) {
      std::cout << "    " << short_desc << "\n";
    }
    if (!long_desc.empty() && long_desc != short_desc) {
      std::cout << "    " << long_desc << "\n";
    }
    print_metric_events(metric);
  }

  uint32_t num_domains = 0;
  check(mcptiDeviceGetNumEventDomains(device, &num_domains),
        "mcptiDeviceGetNumEventDomains");
  std::vector<MCpti_EventDomainID> domains(num_domains);
  size_t domain_bytes = domains.size() * sizeof(MCpti_EventDomainID);
  check(mcptiDeviceEnumEventDomains(device, &domain_bytes, domains.data()),
        "mcptiDeviceEnumEventDomains");

  std::cout << "Events:\n";
  for (MCpti_EventDomainID domain : domains) {
    uint32_t num_events = 0;
    check(mcptiEventDomainGetNumEvents(domain, &num_events),
          "mcptiEventDomainGetNumEvents");
    std::vector<MCpti_EventID> events(num_events);
    size_t event_bytes = events.size() * sizeof(MCpti_EventID);
    check(mcptiEventDomainEnumEvents(domain, &event_bytes, events.data()),
          "mcptiEventDomainEnumEvents");

    for (MCpti_EventID event : events) {
      std::string name = event_string(event, MCPTI_EVENT_ATTR_NAME);
      std::string short_desc =
          event_string(event, MCPTI_EVENT_ATTR_SHORT_DESCRIPTION);
      std::string long_desc =
          event_string(event, MCPTI_EVENT_ATTR_LONG_DESCRIPTION);
      std::string haystack = name + " " + short_desc + " " + long_desc;
      if (!contains_filter(haystack, filter)) {
        continue;
      }
      std::cout << "  " << name << "\n";
      if (!short_desc.empty()) {
        std::cout << "    " << short_desc << "\n";
      }
      if (!long_desc.empty() && long_desc != short_desc) {
        std::cout << "    " << long_desc << "\n";
      }
    }
  }

  return EXIT_SUCCESS;
}
