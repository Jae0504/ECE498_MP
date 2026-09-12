#pragma once

#include <string>

namespace tlmp {

// Call once after all measurements and CSV flushes in an invocation.
// Plot failures are reported without discarding successful measurements.
void update_roofline(const std::string& kernels, const std::string& ceilings,
                     const std::string& output, int roof_threads = 0);

}  // namespace tlmp
