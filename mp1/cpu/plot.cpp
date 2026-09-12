#include "roofline_plot.h"
#include "plot_config.h"

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <iostream>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace tlmp {

void update_roofline(const std::string& kernels, const std::string& ceilings,
                     const std::string& output, int roof_threads) {
  // Activation must also affect plotting launched by these C++ executables,
  // even if the build was originally configured with a system interpreter.
  std::string python = TLMP_PLOT_PYTHON;
  if (const char* environment = std::getenv("VIRTUAL_ENV");
      environment != nullptr && environment[0] != '\0') {
    python = (std::filesystem::path(environment) / "bin" / "python3").string();
  }
  // Pass arguments directly: filenames must never be interpreted by a shell.
  std::vector<std::string> arguments = {
      python, TLMP_PLOT_SCRIPT, "--kernels", kernels,
      "--ceilings", ceilings, "--output", output};
  if (roof_threads > 0) {
    arguments.insert(arguments.end(), {"--roof-threads", std::to_string(roof_threads)});
  }
  std::vector<char*> argv;
  for (auto& argument : arguments) {
    argv.push_back(argument.data());
  }
  argv.push_back(nullptr);

  // Preserve --format csv stdout, including when the Python plotter reports
  // missing inputs or dependencies. All plotting messages go to stderr.
  std::cout.flush();
  std::cerr.flush();
  posix_spawn_file_actions_t actions;
  int error = posix_spawn_file_actions_init(&actions);
  if (error == 0) {
    error = posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO,
                                           STDOUT_FILENO);
    if (error == 0) {
      pid_t child = 0;
      error = posix_spawn(&child, argv[0], &actions, nullptr, argv.data(), environ);
      posix_spawn_file_actions_destroy(&actions);
      if (error == 0) {
        int status = 0;
        pid_t waited;
        do {
          waited = waitpid(child, &status, 0);
        } while (waited == -1 && errno == EINTR);
        if (waited != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
          return;
        }
        if (waited == -1) {
          std::cerr << "warning: cannot wait for Roofline plotter: "
                    << std::strerror(errno) << '\n';
        }
        std::cerr << "warning: automatic Roofline update failed; measurement CSV "
                     "is saved. Re-run scripts/generate_roofline.py after "
                     "resolving the plotting error.\n";
        return;
      }
    } else {
      posix_spawn_file_actions_destroy(&actions);
    }
  }
  std::cerr << "warning: cannot start Roofline plotter with " << python << ": "
            << std::strerror(error) << "; check the active virtual environment "
            << "or CMake Python3_EXECUTABLE. Measurement CSV is saved.\n";
}

}  // namespace tlmp
